<!-- ********************************************************************************
Copyright (c) 2025, Wentao Guo, Mayank Mishra, Xinle Cheng, Ion Stoica, Tri Dao
******************************************************************************** -->

# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations
[![arXiv](https://img.shields.io/badge/arXiv-2512.14080-b31b1b.svg)](https://arxiv.org/abs/2512.14080)

**SonicMoE** is a blazing-fast Mixture-of-Experts (MoE) implementation optimized for NVIDIA Hopper and Blackwell GPUs, leveraging [CuTeDSL](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html) and [Triton](https://triton-lang.org/main/getting-started/tutorials/index.html).

![image](./assets/mem.png)
![image](./assets/tput.png)

## Prerequisites

- NVIDIA Hopper GPUs (H100, H200) or Blackwell GPUs (GB200, B200, B300)
- CUDA 12.9+ (13.0+ for B300)
- Python 3.12+, PyTorch 2.7+
- `USE_QUACK_GEMM=1` for Blackwell kernels

## Default Path & Environment Flags

The default code path is the **fused-v2 frontier** (FP8 epilogue blockscaled quant + fused-gated up-proj + TMA reduce-add wgrad + CUDA fused topk-metadata kernel). Most flags below are already set to their production-recommended defaults; they are documented here so other agents can quickly reason about the active path.

### Sonic-meta routing kernel (always-on)

The fused CUDA topk→metadata kernel (`sonicmoe.ernie_compat.deepep_topk_metadata_cuda`) is **JIT-compiled on first import** and dispatched automatically by `deepep_topk_to_sonic_metadata`. There is **no env flag** to gate it: when the import succeeds it is always used; when the build fails the code falls back to the Triton-only path. To force the legacy path for A/B comparison, delete the build directory before import:

```bash
rm -rf $TORCH_EXTENSIONS_DIR/sonicmoe_deepep_topk_metadata_cuda  # (or matching paddle build dir)
```

### FP8 / Frontier flags

| Env flag | Default | Recommended | What it gates |
|---|:---:|:---:|---|
| `SONIC_MOE_FP8_MODE` | unset | `perf` (or via `enable_fp8()` ctx) | Master switch. `perf` enables the full FP8 frontier; `mem` enables FP8 with stage-wise memory reuse; unset = BF16 path. |
| `USE_QUACK_GEMM` | unset | `1` | Required on Blackwell (B200/B300) to enable the CuTeDSL FP8 GEMMs. Without it the Paddle compat layer falls back to BF16. |
| `SONIC_MOE_FP8_WGRAD` | unset | `1` | Enable FP8 wgrad (otherwise wgrad runs in BF16). Implied when `fp8_wgrad=True` on the config. |
| `SONIC_MOE_FP8_FUSED_GATED` | `1` | `1` | Fuse SwiGLU gate + epilogue blockscaled quant inside the up-proj GEMM. |
| `SONIC_MOE_FP8_EPILOGUE_QUANT` | `1` | `1` | Quantize y2 / z to FP8 inside the GEMM epilogue (vs. a separate quant kernel). |
| `SONIC_MOE_FP8_FUSED_SWIGLU_QUANT` | `1` | `1` | Fuse SwiGLU activation with the row-wise FP8 quant on the y1 path. |
| `SONIC_MOE_FP8_SAVE_Z_FP8` | `1` | `1` | Save z_fp8 (post-up-proj activation in FP8) for the down-proj backward, instead of storing y1 in BF16. |
| `SONIC_MOE_FP8_RECOMPUTE_Z` | `0` | `0` | Skip storing z_fp8 in fwd; rerun up-proj GEMM in down-proj bwd (Option A). Saves ~213 MiB / active layer at ERNIE shape; ~5–15% extra cost per layer. |
| `SONIC_MOE_FP8_RECOMPUTE_OPT_B` | `0` | `0` | **Do not enable.** Experimental quant-only recompute kernel; produces an `illegal-instruction` fault on non-uniform routing (verified). Kept for research. |
| `SONIC_MOE_FP8_FUSED_ZY1_QUANT` | `0` | `0` | Fuse z & y1 quant into a single dual kernel. Off by default — activate only after benchmarking; the separated path is currently faster on production shapes. |
| `SONIC_MOE_FP8_WGRAD_BETA_ACCUM` | `0` | `0` | Fall back to the legacy `D = A@B + 1.0*C` fused beta-accumulation epilogue (86 regs/thread) instead of TMA reduce-add (50 regs/thread). TMA reduce-add is 2–4% faster end-to-end. |
| `SONIC_MOE_FP8_ASSUME_ALIGNED` | `0` | `0` (eager-mode); `1` (full-graph training) | Skip the runtime padding-check H2D sync; assume `TK % 32 == 0`. Required for zero-sync execution but unsafe if any caller can produce mis-aligned token counts. |
| `SONIC_MOE_FP8_BLOCKSCALED_EXPERT_CAPACITY` | unset | unset (use real load) | Override the per-expert capacity used by the blockscaled FP8 GEMM. Only set when comparing fixed-capacity baselines; the default uses the runtime expert load. |
| `SONIC_MOE_FP8_UPPROJ_EPILOGUE_PRECISION` | `fp8-blockscaled` | `fp8-blockscaled` | Output precision of up-proj epilogue. `bf16` = legacy fallback. |
| `SONIC_MOE_FP8_DOWNPROJ_MAINLOOP_PRECISION` | `fp8-blockscaled` | `fp8-blockscaled` | Mainloop precision of down-proj. |
| `SONIC_MOE_FP8_DOWNPROJ_WEIGHT_PRECISION` | `bf16` | `bf16` | Weight precision feeding down-proj. `fp8` requires `..._MAINLOOP_PRECISION=fp8-blockscaled`. |
| `SONIC_MOE_STAGEWISE_MEMORY` | `0` | `0` (perf); `1` (mem) | Free up-proj activations as soon as down-proj consumes them — tradeoff: ~3–5% extra cost for ~1.0–1.5 GB peak savings at ERNIE shape. |
| `SONIC_MOE_CACHE_DIR` | `~/.cache/sonicmoe` | (default) | Override the JIT compile-cache directory used by `jit_warmup`. |

Production single-GPU launcher template (matches `tests/single_card_tests/...test_gpt_model_moe_sonic_moe.py`):

```bash
USE_QUACK_GEMM=1 \
SONIC_MOE_FP8_MODE=perf \
SONIC_MOE_FP8_WGRAD=1 \
TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas \
python -m pytest tests/...
```

## Paddle Integration (ERNIE / PaddleFleet)

The `race-fix-paddle` branch integrates SonicMoE into Paddle via `paddle.compat.enable_torch_proxy`. Production entry point: `SonicMoEMlpNode`.

### Training Loop (S74+ contract — `node.step()` BEFORE `optimizer.step()`)

```python
from sonicmoe.ernie_compat import SonicMoEMlpNode

node = SonicMoEMlpNode(experts, n_experts=E, hidden_size=H, intermediate_size=I)

# ── Cold start: first fwd+bwd triggers JIT compilation (~42s) ──
# For explicit warmup before training:
#   from sonicmoe.jit_warmup import warmup_jit
#   warmup_jit(E=8, H=3072, I=1536, device="cuda")

for step in range(num_steps):
    for mb in microbatches:
        out = node(x, tokens_per_expert, dispatched_indices, dispatched_probs)
        out.backward(grad)
    node.step()           # MUST run BEFORE optimizer.step():
                          # converts native CUTLASS [E,2I,H] → ERNIE split-half
                          # [E,H,2I] in-place into expert.weight.main_grad,
                          # which the optimizer then reads.
    optimizer.step()
    optimizer.clear_grad()
```

`main_grad` is **lazy-allocated on first backward** (S74 follow-up); inference / warmup-only flows pay zero `main_grad` memory. Weight cache invalidation is automatic via `(data_ptr, _inplace_version(w))` keys — no manual `clear_*` needed after in-place optimizer updates.

### Cold Start vs Hot Start

| Phase | What happens | Time |
|-------|-------------|:----:|
| **Import** | CUDA topk metadata kernel compiled | ~4s |
| **1st fwd+bwd** | CuTe CUTLASS GEMM + all Triton kernels JIT compiled | ~42s |
| **2nd fwd+bwd** | All caches hit, steady-state | 0.05s |
| **New seqlen** | CuTe GEMM: **0 recompile** (dynamic dims via `mark_layout_dynamic`). Triton: ~2.5s per new TK (cached in `~/.triton/cache` across sessions) | 0-2.5s |
| **After optimizer.step()** | Call `node.step()` → flushes native-layout wgrad to per-expert `main_grad`, clears weight/FP8/topk caches | <1ms |

### JIT Cache Architecture

Three tiers of caching, each with different invalidation strategies:

| Cache | Key includes | Invalidated by | Max size |
|-------|-------------|---------------|:--------:|
| **CuTe compile cache** (`_COMPILE_CACHE*`) | Static model dims (H, I, E, dtype, tile config) only. **No token counts.** | Never (persistent for model lifetime) | Unbounded (typically 3-8 entries) |
| **Fast-path runtime cache** (`_GEMM_FAST_PATH*`) | Exact problem shape (total_M/K + all tensor dims) | Auto-eviction at 64 entries | 64 |
| **FP8 weight cache** (`_WEIGHT_CACHE` etc.) | `data_ptr + _inplace_version + shape + stride` | `node.step()` / `invalidate_weight_caches()` | 8 per cache |
| **Triton JIT cache** (`~/.triton/cache/`) | Full kernel source hash | `rm -rf ~/.triton/cache` | Unbounded (disk) |

**Design principle**: `compile_key` contains only static model dimensions — never `TK`, `total_M`, `total_K`, `capacity`, or any token-count-dependent value. Dynamic token dimensions are handled at runtime via CuTe's `mark_layout_dynamic`. This ensures **zero CuTe recompilation** when batch size or routing distribution changes.

### Multi-process JIT cache on shared GPFS (production)

Production deploys 1 process per GPU; every rank on every node sees the
same `build/` directory and `SONIC_MOE_CACHE_DIR` over GPFS. The JIT layer
is fully race-safe under concurrent cold start across ranks:

| Layer | Race-safety mechanism |
|---|---|
| **C++ extensions** (`sonicmoe/jit.py`) | Cross-process `FileLock` at `<parent>/.{module_name}.lock` (stable inode survives `build/<module>/` wipe). After lock acquisition the worker first tries `_try_import_prebuilt` — directly `dlopen`-imports `<build_dir>/<name>/<name>.so` (PYBIND11) bypassing paddle's racy `load()` whenever the artifact already exists. Only the first rank actually compiles; later ranks fast-path import. |
| **Triton kernels** (`~/.triton/cache`, or `$TRITON_CACHE_DIR`) | Per-key subdirs; same-key concurrent writes use Triton's atomic temp+rename. |
| **Quack / CuTe autotuner** (`autotuner.py:check_disk_cache`) | `sha256(VERSION + tuning_key + configs)` → per-key file. Different shapes write to different files; same-key writes are atomic. Once any rank has tuned a (dtype, K, config) combination, all other ranks read the same JSON on the next iteration. |
| **`paddle.enable_compat()` ordering** | `_get_cpp_function` re-arms the `torch.utils.hipify` proxy blocker (`install_quack_paddle_compat()`) **inside the FileLock immediately before `cpp_extension.load()`**, so the patches are live regardless of whether the consumer imported sonicmoe before or after `paddle.enable_compat()`. |

Heterogeneous-shape concurrent cold starts (e.g. GPU0 hits `total_K=4096`,
GPU1 hits `total_K=8192` simultaneously on a fresh cache) are explicitly
covered by `tests/ops/test_jit_concurrent_heterogeneous.py` and gated in CI
under phase `jit-concurrent`.

**Recommended deployment**: pre-warm once per cluster, then point every rank
at the shared cache:

```bash
python -m sonicmoe.cli.warmup --E 32 --H 3072 --I 1536 \
    --cache-dir /gpfs/.../sonicmoe_jit_e32_h3072_i1536
# every rank
export SONIC_MOE_CACHE_DIR=/gpfs/.../sonicmoe_jit_e32_h3072_i1536
```

Skipping pre-warm is also safe: ranks will JIT-compile in parallel under the
locks above; the cluster pays the cold-start cost once and converges to
steady-state on iteration ≥ 2.

### Custom integration (PaddleFleet / external trainers)

1. Call `paddle.enable_compat()` (or `paddle.compat.enable_torch_proxy(scope={"sonicmoe", "quack", "triton"})`) **before** `import sonicmoe`. The reverse order is also tolerated — `_get_cpp_function` re-arms patches lazily — but the recommended order avoids a duplicate proxy install on the hot path.
2. Set `SONIC_MOE_CACHE_DIR` to a shared GPFS path so all ranks reuse Triton/Quack disk caches; otherwise each rank caches under `~/.cache/sonicmoe`.
3. Construct one `SonicMoEMlpNode` per fused expert group (see Paddle Integration above). Call `node.step()` BEFORE `optimizer.step()` each microbatch; this flushes native-layout wgrad into per-expert `main_grad`.
4. For diagnostics: `SONIC_MOE_VALIDATE_DISPATCH=1` enables an O(T) per-row uniqueness check on `dispatched_indices` before kernel launch (catches malformed routing in custom dispatchers).

### Lessons learned (S77 strict CI hardening)

- **paddle.distributed.launch on paddlejob: WHITELIST env, never denylist**. The paddlejob cluster exports `PADDLE_TRAINERS=4 IPs`, `DISTRIBUTED_TRAINER_ENDPOINTS` (32 entries), `POD_*`, `EKS_POD_*`, `PADDLE_CURRENT_ENDPOINT`, `PADDLE_CLUSTER_TRAIN`, `PADDLE_IS_LOCAL=0`, etc. Any one of these makes the launcher silently enter multi-NODE rendezvous mode and block forever (no error, no children, launcher in `R` state). `tools/ci/multicard_smoke.py` builds env from a strict prefix whitelist (`PATH/LD_/HOME/USER/LANG/CUDA_/NVIDIA_/TRITON_/SONIC_MOE_/FLAGS_/NCCL_/GLOG_/OMP_/PYTHON…`).
- **`Place(gpu:N) is not supported` is almost always a lazy device-pool init bug**. The pool only registers the place named by `FLAGS_selected_gpus`; any other place errors. Eager-allocate a 1-element tensor right after `paddle.device.set_device(...)` to force pool registration before any async path (autograd backward, `paddle.library` proxies inside quack JIT, `paddle.tensor.random.gaussian`) hits it. Same root cause as the production crash from `quack.autotuner._gpu_warmup`.
- **`_FP8Config()` snapshots `is_fp8_active()` at construction, not at use**. Always construct it INSIDE the `with enable_fp8(True):` block; otherwise a prior test/code path's `enable_fp8(False)` lingering state leaks and the wgrad path silently falls back to BF16. Belt-and-braces: set `SONIC_MOE_FP8_MODE=perf` BEFORE `import sonicmoe`.
- **Multi-process JIT cache on shared GPFS**: `sonicmoe.jit` uses `FileLock` on a stable parent directory (NOT inside `build_directory`, which paddle wipes mid-cycle). Triton + Quack disk caches naturally support multi-rank reuse; sentinel (`{cache_root}/warmup_sentinel.json`) gates cold compile. Cross-rank shape divergence (rank 0 sees shape A, rank 1 sees shape B) is supported because each rank takes the file lock per-key.
- **quack import path**: `/usr/local/bin/python` (the default `python` on this host) lacks the `quack` site-package; it lives at `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack`. `tests/conftest.py` injects this into `sys.path` automatically; `tools/ci/jit_bench.py::_run_subprocess` and `tools/ci/multicard_smoke.WORKER_BODY` do the same for spawned workers.
- **Hipify proxy install order matters**. paddle's torch-proxy intercepts `torch.utils.hipify.hipify_python` lookups and raises `KeyError`. Fix: pre-import the real `torch.utils.hipify[.hipify_python]` and add it to `paddle.compat.extend_torch_proxy_blocked_modules`. Re-arm inside the JIT lock so the patch applies regardless of `import sonicmoe` vs `paddle.enable_compat()` ordering.
- **Blackwell ptxas mismatch**. Bundled Triton 3.5 ptxas does not recognize `sm_103a`. Tests/CI globally export `TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas` (handled in `tests/conftest.py::_ensure_blackwell_ptxas`). Production deployment should set this if running on B300.
- **`dispatched_indices` per-row uniqueness contract**. The kernel (`sonicmoe/ernie_compat/deepep_topk_metadata_cuda/kernel.cu` L315-333) assumes each row of `dispatched_indices` contains pairwise-distinct expert ids. Real DeepEP dispatch always satisfies this; custom test fixtures must. The optional contract validator at `deepep_metadata.py` (gated by `SONIC_MOE_VALIDATE_DISPATCH=1`) catches violations early.
- **paddle's `cpp_extension.load()` writes a wrapper `.py` whose `__bootstrap__` references `*_pd_.so`**. In this paddle build the file is plain `*.so` and the wrapper is non-functional; importing the `.so` directly via `spec_from_file_location` (PYBIND11 handles symbol export) is the correct fast-path.
- **Stable lock paths matter**. paddle's `load()` may wipe its `build_directory` mid-cycle. The FileLock must live in the *parent* dir so the lock inode survives.

### Gradient Contract

| Gradient | Mechanism | Verified |
|----------|-----------|:--------:|
| **dx** (d/d hidden_states) | Paddle autograd through `_SonicMoEDeepEPFunc.backward` | cos=0.9975 |
| **ds** (d/d dispatched_probs) | `_GatherRouterScores` PyLayer with custom Triton scatter (no CUB cascade) | cos=0.9971–0.9973 |
| **dw1, dw2** | CUTLASS wgrad accumulates directly into the per-instance fused `[E, 2I, H]` / `[E, H, I]` native buffer (lazy-allocated on first backward); `node.step()` performs the in-place native→ERNIE split-half layout conversion into `expert.weight.main_grad` before `optimizer.step()` reads it. | cos=0.9975 / 0.9971 |

### Precision (Session 65, FP8 vs BF16 gold, TMA Reduce-Add epilogue)

| N | K | E | I | out | dx | dw1 | dw2 |
|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 128 | 4 | 4 | 384 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 128 | 8 | 8 | 384 | 0.9979 | 0.9975 | 0.9975 | 0.9971 |
| 512 | 4 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 512 | 8 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 1024 | 8 | 8 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9972 |
| 256 | 8 | 32 | 1536 | 0.9979 | 0.9975 | 0.9975 | 0.9971 |

ds gradient verified via `test_cold_start_e2e.py`: cos=0.9972 across all 6 shapes (1024/8192/4096/2048/512/16384 tokens).

All cosine > 0.99, RRMSE < 7.6%.  Shapes include E=32 (production), varying topk (4/8), small/large token counts.

### Performance (nsys GPU-projection, B30Z, TMA Reduce-Add)

Session 65 (`SonicMoEMlpNode`, TMA reduce-add wgrad epilogue, default config):

| Shape (I=1536 K=8) | S53 BF16 (µs) | Paddle FP8 (µs) | Speedup vs BF16 |
|---|:---:|:---:|:---:|
| T=8192 E=8 | 3644 | 2820 | **1.29x** |
| T=8192 E=32 | 3844 | 3283 | **1.17x** |
| T=16384 E=8 | 7953 | 5548 | **1.43x** |
| T=16384 E=32 | 8129 | 5916 | **1.37x** |

TMA reduce-add optimization (Session 65): replaced fused `D=A@B+1.0*C` wgrad epilogue
(86 regs/thread) with TMA hardware atomic add on store (50 regs/thread). Improvement:
- E=8: -65 µs/iter (-2.3%)
- E=32: -138 µs/iter (-4.0%)
- BF16 wgrad GEMM per-call: -16µs (E=8), -33µs (E=32), 5-7.7% faster

To fall back to the legacy fused beta=1.0 epilogue: `SONIC_MOE_FP8_WGRAD_BETA_ACCUM=1`

ERNIE-shape detail (E=32 H=3072 I=1536 K=8 EP=8 SEQ=4096):
- **Forward GPU-proj: 625 µs** (CUTLASS GEMM 65%, FP8 quant 10%, router 14%)
- **Backward GPU-proj: 1904 µs** (wgrad 78%, actgrad 13%, quant 5%)
- **Total: 2530 µs/iter** (CV < 0.3%)

See `HANDOFF.md` for full kernel breakdown and Session 53 baseline comparison.

### Key Files

| File | Purpose |
|------|---------|
| `sonicmoe/ernie_compat/mlp_node_v2.py` | `SonicMoEMlpNode`: production MlpNode with `.forward()`, `.step()`, `.warmup()` |
| `sonicmoe/jit_warmup.py` | `warmup_jit(E, H, I)`: pre-compiles all CuTe + Triton kernels |
| `sonicmoe/quack_utils/blockscaled_fp8_gemm.py` | FP8 GEMM wrappers (CUTLASS + Triton quant), cache key design |
| `sonicmoe/quack_utils/swiglu_triton.py` | Fused SwiGLU Triton kernels (5 production + 2 legacy bf16 variants) |
| `sonicmoe/quack_utils/_validate.py` | Low-overhead input validation (dtype/stride/shape, zero GPU sync) |
| `sonicmoe/ernie_compat/deepep_metadata.py` | DeepEP topk → SonicMoE routing metadata conversion |
| `sonicmoe/functional/__init__.py` | `_UpProjection`, `_DownProjection` autograd Functions |

### Test Files

| Test | What it validates | Run command |
|------|-------------------|-------------|
| `test_cold_start_e2e.py` | Cache clear → JIT → 6-shape precision (out/dx/ds/dw1/dw2) | `CUDA_VISIBLE_DEVICES=2 python tests/ops/test_cold_start_e2e.py` |
| `test_mlpnode_correctness_large.py` | **Topk kernel bug-fix regression** — 9 cases incl. SEQ=16K (TK=131072), skew/extreme/holes | `CUDA_VISIBLE_DEVICES=7 python tests/ops/test_mlpnode_correctness_large.py` |
| `test_jit_optimization.py --quick` | Correctness (cos>0.99), zero JIT recompile, memory | `CUDA_VISIBLE_DEVICES=0 python tests/ops/test_jit_optimization.py --quick` |
| `test_mlpnode_precision.py` | Multi-topk precision audit | `CUDA_VISIBLE_DEVICES=0 python tests/ops/test_mlpnode_precision.py` |
| `bench_mlpnode_mem.py` | E=32 fwd+bwd memory benchmark (ERNIE shape) | `CUDA_VISIBLE_DEVICES=1 python tests/ops/bench_mlpnode_mem.py` |
| `bench_wgrad_epilogue.py` | A/B wgrad epilogue benchmark (TMA add vs fused beta) | `CUDA_VISIBLE_DEVICES=2 python tests/ops/bench_wgrad_epilogue.py` |
| `bench_mlpnode_topk_nsys.py` | nsys GPU-projection benchmark | Wrap with `nsys profile --resolve-symbols=false` |

### Read First (for next developer/agent)

| Priority | Resource | Path |
|:---:|----------|------|
| 1 | **Handoff (current state)** | Root `HANDOFF.md` — top section is the active session, prior sessions preserved verbatim below |
| 2 | **PaddleFleet migration** | `docs/PADDLEFLEET_MIGRATION_S74.md` — stream patch, `node.step()` ordering, lazy main_grad, Fleet's pre-fused-weight integration path |
| 3 | **This README** | Root `README.md` — architecture, cache design, training loop, test matrix |
| 4 | **Knowledge Base** | `docs/KNOWLEDGE_BASE.md` — deep expert reference |
| 5 | **Engineering Log** | `reports/fp8_upgrade/engineering_log.md` — historical lesson log (Phases 1–26 only; superseded by HANDOFF.md sessions ≥66) |
| 6 | **Environment** | `/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/env.md` — machine setup, Paddle compat pitfalls, perf methodology |

> **Project state (as of session 77, 2026-04-29)**: branch `myrepo/race-fix-paddle` (PFCCLab/supersonic-moe), tracks `fork/paddle@108322c`. **`bash tools/ci/run_core_tests.sh` → 13/13 PASS, 0 SKIP** on the paddlejob shared-GPFS host (~15 min wall, 2× B30Z Blackwell). FP8 frontier remains green: precision (out/dx/dw1/dw2/ds) cos≥0.997, multilayer/multistep main_grad accumulation correct, single-stream from deepep-fwd to deepep-bwd, post-warmup zero `cuda.synchronize()`, `node.step()` MUST precede `optimizer.step()`, `main_grad` lazy-allocated. S77 added: cross-test FP8 config isolation, paddlejob cluster-env whitelist for multicard launcher, eager device-pool init to fix `Place(gpu:N) is not supported` under autograd, multi-process JIT cache hardening on shared GPFS.

**Quick-validate the frontier before resuming work**:
```bash
source .runenv.sh
python -m pytest tests/ops/test_mlpnode_multilayer.py tests/ops/test_mlpnode_correctness_large.py \
                 tests/ops/test_colwise_quant.py tests/ops/test_rowwise_quant.py tests/ops/test_fused_quant.py
python tests/ops/test_mlpnode_precision.py     # 6-shape topk precision audit
```

## Native PyTorch Quick Start

```python
import torch
from sonicmoe import MoE, SonicMoEConfig
from sonicmoe.enums import ActivationType

moe = MoE(num_experts=8, num_experts_per_tok=8, hidden_size=3072,
           intermediate_size=1536, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)

x = torch.randn(8192, 3072, device="cuda", dtype=torch.bfloat16)

cfg = SonicMoEConfig(use_fp8=True, use_quack_gemm=True)
with cfg.activate():
    output, aux_loss = moe(x, use_fp8=True)
```

## CI & Pre-commit Hook

Strict-baseline regression runner — every core mechanism (precision, multilayer/PP, quant kernels, JIT cold/warm/reload/reuse, nsys perf, multi-card) is measured and gated against `tools/ci/baselines.json`:

```bash
# fast pre-commit suite (~2 min on warm cache)
bash tools/ci/run_core_tests.sh --fast

# full suite — also runs jit-cold (~10 min), perf gate (nsys), multi-card
bash tools/ci/run_core_tests.sh
```

Install the pre-commit hook once per clone:

```bash
git config core.hooksPath .githooks
```

Offline pre-warm the JIT cache (Triton + Quack disk caches + sentinel) so production training skips the multi-minute first-loss cost:

```bash
python -m sonicmoe.cli.warmup --E 32 --H 3072 --I 1536 \
    --cache-dir /nfs/sonicmoe_jit_e32_h3072_i1536
# then export SONIC_MOE_CACHE_DIR=/nfs/... on every training rank
```

## Citation

```bibtex
@misc{guo2025sonicmoeacceleratingmoeio,
      title={SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations},
      author={Wentao Guo and Mayank Mishra and Xinle Cheng and Ion Stoica and Tri Dao},
      year={2025},
      eprint={2512.14080},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.14080},
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE).
