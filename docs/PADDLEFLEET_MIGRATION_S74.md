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


---

## 7. S77/S78 addendum — distributed-launch hardening & JIT cache contract

> Applies on top of §1–§6. Required reading for any PaddleFleet bump that
> consumes the `myrepo/race-fix-paddle` snapshot (post-S77, commit ≥ `7660ade`).

### 7.1 paddlejob env: **whitelist**, never denylist

`paddle.distributed.launch` reads cluster bootstrap variables out of the
calling process env (`PADDLE_TRAINERS`, `DISTRIBUTED_TRAINER_ENDPOINTS`,
`POD_*`, `EKS_POD_*`, `NCCL_BOOTSTRAP_UID_SOCK_FAMILY`,
`NCCL_SOCKET_IFNAME`, `MASTER_ADDR/PORT`, …). On a paddlejob worker these
are *always* set, and `launch` interprets them as "you are joining a
multi-node rendezvous", silently hanging when no peer ever shows up.

PaddleFleet harnesses that spawn `python -m paddle.distributed.launch` from
within an existing paddlejob slot **must construct the child env from a
whitelist**, not by `os.environ.copy()` minus a denylist. The reference
whitelist used by sonic-moe CI (`tools/ci/multicard_smoke.py`):

```python
ENV_WHITELIST_PREFIX = (
    "PATH", "LD_", "HOME", "USER", "LANG", "LC_", "TERM", "TMPDIR",
    "PWD", "SHELL",
    "PYTHON",                # PYTHONPATH / PYTHONUNBUFFERED / …
    "VIRTUAL_ENV", "CONDA_",
    "CUDA_", "NVIDIA_",      # CUDA_HOME / CUDA_VISIBLE_DEVICES / …
    "TRITON_",               # TRITON_PTXAS_PATH / TRITON_CACHE_DIR
    "SONIC_MOE_", "USE_QUACK_GEMM",
    "FLAGS_",                # paddle FLAGS_*
    "NCCL_", "GLOG_", "OMP_",
)
ENV_DROP_EXACT = ("NCCL_SOCKET_IFNAME", "NCCL_BOOTSTRAP_UID_SOCK_FAMILY")
```

Symptom of regressing this: `paddle.distributed.launch` prints rendezvous
endpoints with the *paddlejob* trainer IP list and never returns. Add a
30-second wall-clock timer around the launch call to fail fast if bumped.

### 7.2 Lazy device-context-pool init (root-cause of S77 production hang)

The original production crash that triggered S77 was:

```
ExternalError: ... Place(gpu:1) is not supported by Executor.
  at paddle/phi/backends/context_pool.cc:81
  in quack/autotuner.py(67) _gpu_warmup → paddle.tensor.random.gaussian
  during GradNodePyLayer SonicMoEDeepEPFunc.backward
```

Root cause: `paddle 3.4`'s `DeviceContextPool` only registers the place
named by `FLAGS_selected_gpus`. Async paths (autograd backward, the
`paddle.library` proxy that quack uses for its custom-op shims, JIT warmup
inside `quack.autotuner._gpu_warmup`) hit the pool with the device
inferred from the current torch-compat thread-local stream, *not* from
`current_device()`.

**Fix contract** that PaddleFleet entry-points must honour:

1. Read `FLAGS_selected_gpus` (or `CUDA_VISIBLE_DEVICES[local_rank]`) and
   call `paddle.device.set_device(f"gpu:{N}")` immediately after worker
   start.
2. Eagerly allocate a 1-element tensor right after the `set_device`:
   ```python
   import paddle
   paddle.device.set_device(f"gpu:{N}")
   _ = paddle.empty([1])           # <- forces DeviceContextPool entry
   ```
3. Only then `import sonicmoe`, instantiate `SonicMoEMlpNode`, or call
   any quack op.

Reference implementation: `tools/ci/multicard_smoke.py:WORKER_BODY`. We
plan to fold this into `sonicmoe._quack_compat` so callers don't have to
remember; until then PaddleFleet must do it explicitly.

### 7.3 `_FP8Config` snapshot timing

`SonicMoEMlpNode.forward` constructs `_FP8Config()` lazily; the constructor
calls `is_fp8_active()` once and snapshots the result. If the node is built
*outside* a `with enable_fp8(True):` block and called *inside* one, the
snapshot is stale and the node silently runs BF16.

```python
# WRONG — config snapshot before fp8 context activates
node = SonicMoEMlpNode(...)
with enable_fp8(True):
    out = node(...)        # runs BF16

# RIGHT — refresh inside the context
with enable_fp8(True):
    node = SonicMoEMlpNode(...)
    out = node(...)
# Or call node._refresh_fp8_config() inside the with-block.
```

`mlp_node_v2.py:722-728` performs the refresh internally on the first
forward inside an `enable_fp8(True)` block, but PaddleFleet harnesses that
hold a long-lived node across enable/disable transitions must call
`node._refresh_fp8_config()` themselves.

### 7.4 Multi-process JIT cache on shared GPFS

In production every GPU is a separate process, *and* the JIT cache lives on
shared GPFS. Three facts the cache layer enforces (do not bypass):

* Cache key includes Triton/CuTe source hash + shape signature
  (`H`, `I`, `E`, `total_K`, dtype) **but not** `T`/`seqlen`. T-axis
  variation must not retrigger compilation. Verified by
  `tests/ops/test_jit_key_stability.py`.
* Disk writes go through `fcntl.flock` on a per-key `.lock` file in the
  cache dir. Cold compile budget covers `flock` contention up to
  ~32 ranks on a single GPFS volume; for larger pods stage a per-node
  cache dir via `SONIC_MOE_CACHE_DIR=/dev/shm/sonic_moe_cache_$LOCAL_RANK`
  and seed it from GPFS once per node.
* **Triton autotune *results* are now disk-cached too** (S78). Without
  this, `token_gather_sum_kernel` alone re-ran a 21 000-launch sweep on
  every cold process (~30 s wall). `sonicmoe._triton_autotune_persist`
  flips `TRITON_CACHE_AUTOTUNING=1` at `import sonicmoe` time so every
  sonic-moe (and quack) `@triton.autotune` kernel persists its best-config
  selection to `$TRITON_CACHE_DIR/<sha>/<kernel>.autotune.json`. Verified
  saving on this host: cold→warm wall **236.9 s → 173.7 s (−63 s/process,
  −27 %)** with bench µs/iter unchanged. Multi-process safe (Triton uses
  atomic-rename writes). Opt-out: `SONIC_MOE_NO_TRITON_AUTOTUNE_CACHE=1`.

PaddleFleet harness expectations:

| Env / call                          | Required value / behaviour |
| ----------------------------------- | -------------------------- |
| `SONIC_MOE_CACHE_DIR`               | shared GPFS dir reachable by every rank; created by rank 0 ahead of `import sonicmoe` |
| Cold-warmup budget per rank         | ≤ 600 s (ptxas + autotune); see `tools/ci/baselines.json::jit.cold_warmup_s` |
| Warm sentinel skip                  | ≤ 5 s; rank N must observe rank 0's `warmup_sentinel.json` after barrier |
| Cross-process reload                | ≤ 300 s (disk-cache hit, no ptxas) |
| In-process dispatch                 | ≤ 8 ms median fwd+bwd (`jit.in_process_reuse_us`) |

If the harness deletes the cache dir between runs, expect the full 600 s
cold path on the next launch; bake `warmup_jit_parallel(workers=N)` into
the harness boot or accept the regression.

### 7.5 `quack` import path (production-only landmine)

`/usr/local/bin/python` does **not** have `quack` site-packaged. The
project relies on injecting
`/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack`
into `sys.path` before any sonic-moe import.

PaddleFleet harnesses that subprocess into `/usr/local/bin/python`
(perf gates, multicard launchers, JIT bench) must propagate either:

* `PYTHONPATH=…/sonicmoe_for_ernie/quack:$PYTHONPATH` in the child env, **or**
* run the child via `eb_venv/bin/python` (already has quack installed):
  `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv/bin/python`.

Reference: `tools/ci/jit_bench.py::_run_subprocess` and
`tests/conftest.py` both inject the path.

### 7.6 Coverage-collection contract (S78)

The CI script `tools/ci/run_core_tests.sh` runs every phase under
`coverage run --source=sonicmoe` (NOT `--append` — incompatible with
`.coveragerc::parallel = True`, see S78) and gates on
`baselines.json::coverage.target_pct` (currently 28 %, set to current
actuals). For PaddleFleet:

* **No source change** required — coverage is process-scoped, won't leak
  into Fleet's harness.
* The CI uses parallel-mode coverage (`.coveragerc::parallel = True`) so
  per-xdist-worker `.coverage.*` files are merged via `coverage combine`.
* Bumping the target requires editing `baselines.json` *and* explaining
  the new floor in HANDOFF — no silent loosening.

### 7.7 Removed/moved symbols since S74

| Old (S74)                                            | New (S77/S78) |
| ---------------------------------------------------- | -------------- |
| `sonicmoe.ernie_compat.mlp_node_v2._W_CACHE`         | per-instance `node._w_cache` (already in S74 docs; reaffirmed) |
| `sonicmoe.warmup.warmup_jit(force=True)`             | unchanged, but now backed by file-locked GPFS-safe writer |
| ad-hoc per-test cache dirs                           | unified `SONIC_MOE_CACHE_DIR`, default `<repo>/.jit_cache` |
| (none)                                               | new `tests/ops/test_jit_key_stability.py` — verifies T-axis non-keying |
| (none)                                               | new `tests/ops/test_jit_concurrent_heterogeneous.py` — verifies cross-rank heterogeneous shape compile under one cache dir |
| (none)                                               | new `tests/ops/test_mlpnode_extreme_shapes.py` — 0-size + extreme-imbalance |
| `tools/ci/multicard_smoke.py`                        | env-whitelist + eager device-pool + `FLAGS_selected_gpus` pin + `TRITON_PTXAS_PATH` |

---

## 8. Validation snapshot accompanying this doc

Captured on B30Z, paddlejob shared GPFS host, post-S77 commit `7660ade`:

| Suite                                      | Result               |
| ------------------------------------------ | -------------------- |
| `bash tools/ci/run_core_tests.sh`          | ✅ 13/13 PASS, 0 SKIP |
| Per-iter perf (FP8, T=8192 H=3072 I=1536 E=8 K=8) | 2519 µs/microbatch (grad_acc=8); GPU-projection 2823 µs/iter mlpnode-only |
| Speedup vs. BF16                           | 1.29×–1.70× (mean 1.53×) |
| Precision (cosine)                         | ≥ 0.997 on `out`, `dx`, `ds`, `dw1`, `dw2` |
| Memory (FP8)                               | +4.8 % – +10.3 % over BF16; mem-mode −24.5 % |

The accompanying nsys timeline at the canonical Ernie shape lives at
`reports/ernie_shape_nsys_s78/trace.nsys-rep` (+ `trace.sqlite`); open in
Nsight Systems 2026.2 or use `tools/parse_nsys_per_iter.py` to dump
per-iter GPU-projection.
