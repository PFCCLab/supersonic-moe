# PaddleFleet Migration — Sonic-MoE Session 74

This note covers the **two production-affecting changes** that PaddleFleet
must understand when bumping to the post-S74 sonic-moe snapshot:

1. **Triton stream compat patch** — auto-installed; *no* PaddleFleet code change
   needed, but understand the implications.
2. **`SonicMoEMlpNode` globals/FIFO purge** — PaddleFleet's current entry-point
   (`run_sonic_moe` calling `_UpProjection.apply` / `_DownProjection.apply`
   directly) is **unaffected**. Only matters if/when PaddleFleet adopts the
   higher-level `SonicMoEMlpNode` wrapper.

It also lists the deleted/renamed symbols so any out-of-tree PaddleFleet test
that imported them gets a clean migration recipe.

---

## 1. Triton ↔ Paddle stream compat (CRITICAL CORRECTNESS FIX)

### Background

Triton's `GPUDriver.get_current_stream` is bound to the C-level
`torch._C._cuda_getCurrentRawStream` at import time
(`triton/backends/driver.py`). That C function bypasses any Python-side
`paddle-torch-compat` shim and **always returns PyTorch's NULL stream
(`0x0`)**.

Result before this patch:

| Op family            | CUDA stream observed in nsys (run #7) |
| -------------------- | ------------------------------------- |
| Paddle GEMM / phi::* | stream **13** (Paddle compute stream) |
| CUTLASS quack GEMMs  | stream **13** (uses `_get_raw_cuda_stream` → paddle proxy) |
| **Sonic-MoE Triton kernels** (quant / gather / scatter) | **stream 7 = legacy NULL** |

Stream 7 is the CUDA NULL stream (per-context default). NULL-stream launches
have **implicit cross-stream synchronization**: every kernel on stream 7
serialises against every other stream, and producer/consumer pairs that
straddle stream 7 ↔ stream 13 race unless explicitly synchronised.

### Fix

`sonicmoe/_triton_stream_compat.py` monkey-patches
`triton.runtime.driver.driver.active.get_current_stream` to return
`paddle.device.current_stream().stream_base.raw_stream` instead. The patch is
installed at the very top of `sonicmoe/__init__.py`, so it takes effect the
moment `import sonicmoe` (or `import paddlefleet.ops.sonicmoe`) runs — i.e.
strictly before the first Triton kernel can launch.

Verified post-patch:

```
triton stream: 0x5b5366aec7c0   ← Paddle compute stream
paddle stream: 0x5b5366aec7c0
torch  stream: 0x0              ← (unchanged; sonic-moe never relies on this)
```

### Behaviour for PaddleFleet

* **Zero code change required.** PaddleFleet imports `paddlefleet.ops.sonicmoe`
  during module load, which transitively triggers the patch.
* The patch is idempotent and re-entrant safe.
* Opt-out for debugging: `SONIC_MOE_NO_TRITON_STREAM_PATCH=1` env var.
* Falls back to the original `torch._C._cuda_getCurrentRawStream` if the
  paddle import / `current_stream()` call raises — never breaks a non-paddle
  caller.

### Validation checklist for PaddleFleet integrators

After bumping the sonic-moe snapshot, sample a fresh nsys timeline of any FP8
training step and verify:

1. `_quantize_and_pack_kernel`, `_quantize_pair_kernel`,
   `_colwise_quantize_and_pack_kernel`, `token_gather_sum_kernel`,
   `_gather_isa_packed_scales_kernel`, `_scatter_router_grad_kernel`
   all appear on the **same stream** as `kernel_cutlass_kernel_quackgemm_*`.
2. The legacy NULL stream (typically displayed as the lowest-numbered stream)
   contains **no sonic-moe kernels**.

If either check fails, confirm `sonicmoe._triton_stream_compat._PATCHED` is
`True` after `import paddlefleet`.

---

## 2. `SonicMoEMlpNode` globals/FIFO purge

### What changed

The `SonicMoEMlpNode` wrapper was rewritten to **own all per-instance state**:

* Module-level `_W_CACHE` dict and `_PENDING_FLUSH_LAYERS` FIFO list **removed**.
* `_NATIVE_W1_GRAD`, `_NATIVE_W2_GRAD`, `_NATIVE_GRAD_EXPERTS`, `_NATIVE_GRAD_I`
  globals **removed**.
* `_SonicMoEDeepEPFunc._topk` class-variable hack **removed** — `topk` is now a
  regular forward arg.
* `_ensure_native_grads`, `_accumulate_w1`, `_accumulate_w2`, `_mark_pending_flush`,
  `flush_native_grads()` (production callers) **removed**.
* Legacy `SonicMoEFunc` PyLayer + `prepare_sonic_inputs` helper **deleted**.

In their place, every `SonicMoEMlpNode` instance owns:

* `self._w_cache: dict` — stacked-weight reuse across iters of the *same* layer.
* `self._pending_flush: bool` — set by ctx in backward, cleared by `step()`.
* `self._warmed_for_step: bool` — JIT/cache warmup gate per global step.

New public API surface:

| Method                      | Purpose                                                   |
| --------------------------- | --------------------------------------------------------- |
| `node.forward(...)`         | unchanged                                                 |
| `node.step()`               | flushes wgrads (native CUTLASS layout → ERNIE split-half layout) into `expert.weight.main_grad`. **MUST be called BEFORE `optimizer.step()`** — the optimizer reads the same storage that `step()` writes. |
| `node.flush_grads()` *alias*| same as `node.step()` (kept for harness compatibility)    |
| `node.invalidate_caches()`  | optional: drop per-instance `_w_cache` + FP8 weight-cache entries (only needed under memory pressure; cache keys are version-tagged so in-place optimizer updates auto-invalidate) |

#### `node.step()` ordering contract

```python
# CORRECT: flush native→ernie wgrad layout BEFORE optimizer reads main_grad
loss.backward()
for node in all_sonic_nodes:
    node.step()           # writes into expert.weight.main_grad (in-place)
optimizer.step()          # reads expert.weight.main_grad
optimizer.clear_grad()    # zeros main_grad for next iter
```

The previous (S73) docstring said "after `optimizer.step()`" — that was wrong;
flushing AFTER the optimizer would mean the optimizer applied a wrongly-laid-out
grad. The fix has been applied in code; please mirror the ordering in any
PaddleFleet harness that calls into `SonicMoEMlpNode`.

#### Lazy `main_grad` allocation (memory saver)

`expert.weight.main_grad` (shape `[E, H, 2I]` for w1, `[E, H, I]` for w2) is now
**allocated on first backward**, not on first forward. Inference / warmup-only
flows pay zero `main_grad` memory. Allocation is idempotent and version-safe:
`_alloc_main_grad_w{1,2}` only allocates when the per-expert main_grad is
absent. No behaviour change for training; only a memory savings (~tens of MiB
at small shapes, hundreds at production shapes).

### Impact on the current PaddleFleet integration

`paddlefleet/transformer/moe/moe_layer.py::run_sonic_moe` calls
`_UpProjection.apply` / `_DownProjection.apply` directly. It does **not**
import `SonicMoEMlpNode`, `flush_native_grads`, or any of the deleted globals.

> **Conclusion: the current PaddleFleet `using_sonic_moe` path needs zero
> source changes.** Just bump the sonic-moe pin and re-run CI.

### Future migration path (optional)

When PaddleFleet wants to consume the high-level wrapper:

```python
from paddlefleet.ops.sonicmoe.ernie_compat.mlp_node_v2 import SonicMoEMlpNode

# Per-layer (instantiate once, reuse across micro-batches & iters):
self.sonic_node = SonicMoEMlpNode(
    experts=self.grouped_gemm_experts,
    n_experts=self.num_experts_per_device,
    hidden_size=H,
    intermediate_size=I,
)

# Per-forward:
y = self.sonic_node.forward(
    dispatched_hidden_states,
    tokens_per_expert,
    dispatched_indices=topk_indices.cast("int32"),
    dispatched_probs=topk_scores,
)

# Per-optimizer-step (after all micro-batches have flushed grads):
self.sonic_node.step()
```

Pipeline-parallel + interleaved 1F1B is supported out of the box because each
layer's node carries its own `_pending_flush` flag — there is no global FIFO
that can be poisoned by an out-of-order F/B sequence.

For schedules where you want to flush grads **without** invalidating the
weight cache (e.g. cross-microbatch grad accumulation), call
`node.flush_grads()` and only call `node.step()` at optimizer-step
boundaries.

### Removed symbols / migration recipes

| Old                                 | New                                       |
| ----------------------------------- | ----------------------------------------- |
| `from sonicmoe.ernie_compat import SonicMoEFunc`      | (deleted; use `SonicMoEMlpNode` or call `_UpProjection`/`_DownProjection` directly as PaddleFleet already does) |
| `from sonicmoe.ernie_compat import prepare_sonic_inputs` | (deleted; use `deepep_topk_to_sonic_metadata` — same import path PaddleFleet already uses) |
| `flush_native_grads()` (module fn)  | `node.flush_grads()` per `SonicMoEMlpNode` instance |
| `_mlp_module._W_CACHE.clear()`      | `node.step()` (clears the per-instance cache) |
| `_mlp_module._PENDING_FLUSH_LAYERS` | `node._pending_flush` (per-instance bool) |
| `_mlp_module._NATIVE_W1_GRAD`/etc.  | (gone — no replacement needed; `_pending_flush` covers the use case) |

The module-level `flush_native_grads` and `stack_ernie_w1`/`stack_ernie_w2`
shims are **kept** but operate on a separate `_LEGACY_W_CACHE` and
`_LEGACY_PENDING_FLUSH` list — they are documented as legacy and used only
by `jit_warmup.py` + a couple of standalone benchmark scripts. Production
`SonicMoEMlpNode` instances do not feed into them.

---

## 3. Router-scores backward — CUB cascade replaced

`_differentiable_router_scores` previously did
`gathered = dispatched_probs.reshape(-1)[gather_idx]`. Paddle dispatched the
backward through generic advanced-indexing, which spawned per-call:

* `cub::DeviceRadixSortHistogramKernel`
* `cub::DeviceRadixSortExclusiveSumKernel`
* `cub::DeviceRadixSortOnesweepKernel` (×3)
* `IndexingBackwardKernel<float, 4>`
* `histogram_kernel<16>`, `prefix_sums_kernel`, `block_offset_scan_kernel`,
  `scatter_and_fixup_kernel<16>`

Total ≈ 0.3–0.5 ms per backward at production shape (T=8192, K=8). This is
unnecessary — `gather_idx` is a permutation of distinct positions, so backward
is a plain scatter (no accumulate, no sort).

S74 replaces the indexing op with `_GatherRouterScores` (custom autograd
function) whose backward is a single Triton kernel
`_scatter_router_grad_kernel`. Bit-exact verified vs. the old path on
`test_mlpnode_*` regression suite.

PaddleFleet integrators get this for free — `_differentiable_router_scores`
is internal to `mlp_node_v2.py` and only runs when using `SonicMoEMlpNode`.
The current `run_sonic_moe` path that builds `scores_for_down` manually is
unaffected.

---

## 4. Validation matrix run before this snapshot

| Suite                                           | Result        |
| ----------------------------------------------- | ------------- |
| `tests/ops/test_mlpnode_precision.py`           | ✅ 1 passed   |
| `tests/ops/test_mlpnode_multilayer.py`          | ✅ 2 passed   |
| `tests/ops/test_mlpnode_correctness_large.py`   | ✅ 1 passed   |
| `tests/ops/test_colwise_quant.py`               | ✅ 32 passed  |
| `tests/ops/test_rowwise_quant.py`               | ✅ 45 passed  |
| `tests/ops/test_fused_quant.py`                 | ✅ 14 passed  |

All bit-exact relative to S73 baseline (commit `2795dc0`).

---

## 5. Quick smoke-test recipe for PaddleFleet CI

```python
import paddlefleet.ops.sonicmoe as _sm  # triggers stream patch
from triton.runtime.driver import driver
import paddle

# 1. Stream patch is active.
assert getattr(driver.active, "_sonic_moe_paddle_patched", False), (
    "sonic-moe Triton stream patch failed to install"
)

# 2. Triton sees Paddle's stream, not NULL.
assert (
    driver.active.get_current_stream(0)
    == paddle.device.current_stream().stream_base.raw_stream
), "Triton stream != Paddle compute stream"
```

Add this as a one-line invariant inside whichever existing PaddleFleet
sonic-moe smoke test runs first; failure indicates the bump regressed the
patch wiring.

---

## 6. Adopting `SonicMoEMlpNode` with Fleet's pre-fused weights

Fleet's `GroupedMLPExpert` (in `paddlefleet/transformer/moe/moe_expert.py`)
already stores **fused** parameters:

```python
# using_sonic_moe branch
self.weight1 = paddle.create_parameter(shape=[E, 2I, H], ...)  # native CUTLASS layout
self.weight2 = paddle.create_parameter(shape=[E, H,  I], ...)
```

`run_sonic_moe` then `permute([1,2,0])` them into `[2I,H,E]` / `[H,I,E]`
logical views and feeds `_UpProjection.apply` / `_DownProjection.apply`.
PyLayer.backward returns the wgrads as positional outputs and Paddle
auto-aggregates them into `weight1.main_grad` / `weight2.main_grad`. **Zero
intermediate buffer, zero stack op, zero per-expert aliasing.** Stays on this
path — it is strictly more efficient than `SonicMoEMlpNode` for the
pre-fused-weight case, and it is unaffected by the S74 refactor.

### When (if ever) to switch to `SonicMoEMlpNode`

The `SonicMoEMlpNode` wrapper exists to serve callers that hold a
**Python `list` of per-expert `nn.Module`s** (e.g. ERNIE training script). Its
constructor today reads `experts[i].up_gate_proj.weight` /
`experts[i].down_proj.weight` and stacks/permutes them into the same logical
view the `_UpProjection` / `_DownProjection` PyLayers expect. For Fleet, this
would mean:

1. Either re-wrap `weight1[E,2I,H]` into `E` fake-expert objects (wasteful;
   forces a stack the first time; aliases per-expert into the fused buffer).
2. Or extend `SonicMoEMlpNode` with a "pre-fused" constructor mode that takes
   `weight1`/`weight2` directly, skips the stack, and writes wgrads straight
   into `weight1.main_grad` / `weight2.main_grad`.

**Recommendation: don't bother.** Option (1) is pure overhead and option (2)
buys nothing that `_UpProjection.apply` direct call doesn't already provide.
The Node wrapper's value-add is the per-instance `_pending_flush` /
router-scatter / lazy main_grad story for the **list-of-experts** layout —
none of which Fleet needs since it already owns the fused buffer.

### Migration checklist (if Fleet adopts the Node anyway)

If at some point Fleet wants the higher-level wrapper (e.g. to share
testing/benchmark harness), here is the contract:

| Concern                          | Fleet adapter requirement |
| -------------------------------- | ------------------------- |
| Expert param layout              | wrap each `weight1[e]`/`weight2[e]` slice as `expert.up_gate_proj.weight` / `expert.down_proj.weight`. Shapes: `[H, 2I]` and `[I, H]` (paddle linear `[in,out]`). |
| `weight1.main_grad` aggregation  | `_alloc_main_grad_w{1,2}` allocates `[E,H,2I]`/`[E,I,H]` and aliases per-expert `.main_grad` views into it. **The fused `weight1.main_grad` itself becomes the slice-of-the-bigger-buffer view** — make sure Fleet's optimizer reads `expert.weight.main_grad` (not the fused tensor's main_grad). |
| `node.step()` ordering           | MUST run BEFORE `optimizer.step()`. See §2 above. |
| Lazy main_grad                   | First backward triggers allocation; inference / warmup-only flows pay zero. |
| Cache invalidation               | Automatic via `(data_ptr, _inplace_version(w))` — no manual `clear_*` needed. |
| Pipeline parallelism             | Per-instance `_pending_flush` carries layer identity through arbitrary F/B interleaving — safe out of the box. |

### Direct-PyLayer path (current Fleet integration)

These bullets summarise what Fleet **must** do (and what it already does) on
the direct `_UpProjection.apply` / `_DownProjection.apply` path:

* Pass `stream_id = paddle.device.cuda.current_stream().cuda_stream` —
  already done. The S74 stream patch makes Triton honour the same stream.
* `s_scatter_idx.stop_gradient = True` and other metadata stop_gradient flags
  — already done.
* `expert_frequency_offset` is `[E+1]` cumulative; `tokens_per_expert` is
  `[E]`; `num_activated_expert_per_token_offset` is `[E]` — output of either
  `general_routing_router_metadata` or `deepep_topk_to_sonic_metadata`. Both
  helpers are async; no HtoD sync.
* `weight1.permute([1,2,0])` / `weight2.permute([1,2,0])` materialise
  `[2I,H,E]` / `[H,I,E]` logical views with stride-only changes (no copy).
  The `_UpProjection` / `_DownProjection` PyLayers and the underlying
  CUTLASS GEMMs accept these.
* `weight1.main_grad` / `weight2.main_grad` aggregation is handled by
  Paddle's PyLayer machinery from the wgrad positional outputs — no manual
  aliasing needed.
* No `node.step()` call needed; no native→ERNIE layout conversion since the
  parameter layout already matches the CUTLASS write layout (`[E, 2I, H]` /
  `[E, H, I]`).

