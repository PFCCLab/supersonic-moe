# DeepEP -> SonicMoE Metadata Gap Breakdown And Experiment Matrix

Date: 2026-07-09

This document is the PR-facing handoff for the current A35B/EB51 SonicMoE
performance investigation.  It records the conclusions that should constrain
future work, the measured bubble breakdown, and the experiment/test matrix
needed to avoid making performance changes that are not bit-exact or not
reproducible.

## Scope

Profiles analyzed:

| purpose | path |
|---|---|
| baseline | `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb51_A35B_768B_logs/nsys_profiles/fp8_baseline` |
| Sonic latest BF16-wgrad run | `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb51_A35B_768B_logs/nsys_profiles/fp8_sonic_update_bf16_wgrad` |
| earlier Sonic fused-convert run | `/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb51_A35B_768B_logs/nsys_profiles/fp8_sonic_fused_convert` |
| exported SQLite for this gap study | `artifacts/metadata_gap_20260709/sqlite/fp8_sonic_update_bf16_wgrad{0..7}.sqlite` |

Important scope facts:

- The production dispatcher is DeepEP: `moe_token_dispatcher_type='deepep'`.
  This is not the MoE allgather path.  Small ordinary NCCL AllGather activity is
  not material to the current gap.
- Barrier-off is not a valid optimization target.  The correctness baseline is
  barrier-on.
- Cross-node Nsight Systems reports must be aligned with absolute time:

```sql
abs_start_ns = TARGET_INFO_SESSION_START_TIME.utcEpochNs + event.start;
abs_end_ns   = TARGET_INFO_SESSION_START_TIME.utcEpochNs + event.end;
```

Do not compare per-report local zero points directly.

## Current Conclusions

| id | conclusion | evidence | consequence |
|---:|---|---|---|
| C1 | Sonic compute is not the current P0 blocker after BF16 wgrad tuning. | In `fp8_sonic_update_bf16_wgrad`, baseline DeepGEMM BF16 wgrad is `75.687 GPU-s / 1.973 ms avg`, Sonic quack BF16 wgrad is `73.701 GPU-s / 1.915 ms avg`. | Further BF16 wgrad tuning is not the next highest-leverage work. |
| C2 | The latest analyzed timeline still did not use the fused metadata+scale path. | The timeline kernel is `_gather_raw_scales_1x32_to_isa_kernel`; the local fused path emits `pack_raw_scales_from_gather_kernel<...>`. | First verify the real training run reaches `deepep_topk_to_sonic_metadata_with_scales()` before judging the fused implementation. |
| C3 | The up-projection path has a stable host-side bubble before the Sonic GEMM launch. | Across 8 reports, `_gather_raw_scales_1x32_to_isa_kernel end -> GemmGatedSm100ZeroMat* launch API start` has p50 `484.6 us`. | The scale-pack kernel itself is not the bottleneck; the wrapper/carrier path is. |
| C4 | The down-projection analogous path is not a metadata bubble. | `other -> scale_pack -> sonic_down_gemm` total p50 is about `16 us`. | Focus on DeepEP dispatch -> up-projection metadata/carrier path first. |
| C5 | `empty`, `view_dtype`, and `transpose` are stable same-launch-thread costs. | Same-thread NVTX shows `view_dtype` p50 `75.8 us`, `empty` p50 `57.7 us`, `transpose` p50 `56.2 us` before each up GEMM launch. | These are valid optimization targets; they are not random cross-thread noise. |
| C6 | The visible NVTX ranges do not explain the entire prelaunch bubble. | Visible same-thread Paddle wrapper work accounts for roughly `190-210 us` order inside a `~485 us` p50 window. | A specialized CUTE/QuACK fast path is needed to recover the remaining uninstrumented Python/CUTE wrapper time. |
| C7 | Large output carriers are a lifetime problem, not merely an allocator micro-optimization. | A35B-like `TK ~= 928768, I=3584` needs about `10.3 GB` live fp8/scale carrier storage for z/y1 and scales. | A naive global scratch buffer is unsafe; reuse must be autograd/lifetime-aware. |
| C8 | The older large end-to-end slowdown was dominated by layout/copy/sync and distributed skew, not by allgather or explicit barrier edits. | Earlier fused-convert analysis showed `DeepEPDispatch backward +99.544s`, `cudaDeviceSynchronize +175.827s`, layout roundtrip kernels `+74.164s`; allgather was not active. | Do not reintroduce barrier-off or allgather-centered explanations.  Remove skew sources before the barrier instead. |

## Bubble Breakdown

The critical sequence in the latest `fp8_sonic_update_bf16_wgrad` timeline is:

```text
scatter_and_fixup_kernel
  -> host/Python gap
_gather_raw_scales_1x32_to_isa_kernel
  -> Sonic up GEMM prelaunch gap
GemmGatedSm100ZeroMat*
```

Per exported report:

| report | count | scatter->scale gap p50 | scale kernel p50 | scale->up gap p50 | total p50 |
|---:|---:|---:|---:|---:|---:|
| 0 | 2240 | 262.7 us | 13.1 us | 485.1 us | 767.4 us |
| 1 | 2560 | 289.2 us | 13.0 us | 516.1 us | 823.6 us |
| 2 | 2560 | 273.8 us | 13.1 us | 509.6 us | 800.2 us |
| 3 | 2560 | 295.9 us | 13.0 us | 543.2 us | 858.2 us |
| 4 | 2560 | 262.9 us | 13.0 us | 492.6 us | 776.1 us |
| 5 | 2560 | 257.8 us | 13.0 us | 488.9 us | 765.0 us |
| 6 | 2560 | 272.0 us | 13.0 us | 494.7 us | 784.3 us |
| 7 | 2240 | 260.3 us | 13.1 us | 496.3 us | 774.3 us |

Report 0 GPU coverage inside `scatter_and_fixup end -> up GEMM start` is only
about `1.7%`; this is real GPU idle, not useful overlap.

Same-launch-thread NVTX for:

```text
_gather_raw_scales_1x32_to_isa_kernel end
  -> GemmGatedSm100ZeroMat* launch API start
```

| item | p50 per up GEMM | p90 | stable count per up GEMM | interpretation |
|---|---:|---:|---:|---|
| full prelaunch window | 484.6 us | 549.1 us | 1 | true host-side bubble before GEMM launch |
| `view_dtype` | 75.8 us | 101.3 us | 15 | dtype/storage reinterpret wrappers, mostly FP8/E8M0 to uint8 for CUTE |
| `empty` | 57.7 us | 77.6 us | 12 | output/scale carrier tensor construction |
| `transpose` | 56.2 us | 75.2 us | 10 | view-only layout permutation, mainly static weight B in the varlen path |
| allocator allocate | 7.2 us | 9.9 us | 6 | visible allocator bookkeeping inside `empty` |

These NVTX categories are nested and are not exact exclusive wall time.  They
still identify stable repeated work on the same CPU thread that later launches
the up GEMM.

## Code Anchors

| target | file/function | current behavior |
|---|---|---|
| DeepEP topk metadata + scale pack | `sonicmoe/ernie_compat/deepep_metadata.py::deepep_topk_to_sonic_metadata_with_scales` | Fused helper returns the normal metadata tuple plus packed scales when the CUDA extension supports it. |
| C++ scale-pack launcher | `sonicmoe/ernie_compat/deepep_topk_metadata_cuda/kernel.cu` | Default tile-pack path is fastest among tested scale-pack variants. |
| up-projection FP8 scale consumption | `sonicmoe/functional/__init__.py::_fused_blockscaled_gated_forward` | Accepts `x_fp8_pre=(x_fp8, x_scales_t, x_scales_tk_pre)` and uses `x_scales_tk_pre.view(_E8M0_DTYPE)`. |
| large carrier allocation | `sonicmoe/quack_utils/gemm_interface.py::gemm_gated` | Allocates `preact_out` and `postact_out` with `torch.empty` when not provided. |
| generic QuACK/CUTE launch wrapper | `sonicmoe/quack_utils/gemm_gated.py::gemm_gated` | Revalidates shapes/layouts, permutes B, builds CUTE tensors, epilogue args, scheduler args, varlen args, and compile key per call. |
| repeated FP8/E8M0 storage view | `sonicmoe/quack_utils/_gated_epilogues.py::_make_cute_tensor_dynamic` | Calls `tensor.detach().view(torch.uint8)` for runtime FP8 tensors. |
| repeated B layout view | `quack/gemm_wrapper_utils.py::GemmWrapperBase.permute_tensors` | In varlen-M, applies `B.permute(1, 2, 0)` every call. |

## Carrier Size At A35B-Like Shape

For the production-like shape used by the scale-pack benchmark
(`TK ~= 928768`, `I = 3584`, `2I = 7168`):

| carrier | dtype | shape | approximate storage |
|---|---|---|---:|
| `preact_out` / z | fp8 | `TK x 2I` | 6.65 GB |
| `postact_out` / y1 | fp8 | `TK x I` | 3.33 GB |
| `z_scale_out` | uint8 | `TK x (2I/32)` | 208 MB |
| `postact_scale_out` | uint8 | `(TK/128) x (I/128) x 512` | 104 MB |

`preact_out`/z and `postact_out`/y1 can be consumed by backward or downstream
paths.  Any reuse scheme must prove that a buffer is not overwritten before all
autograd users are done.

## Can `view`/`transpose` Be Hoisted To Global Step?

Partially.  The safe boundary is tensor storage lifetime, not the textual
operation name.  A tensor view captures a data pointer, storage offset, shape,
stride, and dtype interpretation.  A cached view can be reused only while all of
those remain valid.

| source | lifetime | can hoist to global step? | best optimization |
|---|---|---:|---|
| FP8 weight `B.mT` and low-level `B.permute(1,2,0)` | static for all microbatches until optimizer updates the weight | yes | Build the final CUTE-facing B layout once when the per-step FP8 weight cache is populated, then route a gated fast path that skips generic `B.mT`/`permute_tensors`. |
| weight scale dtype/storage view | static for all microbatches until optimizer updates the weight | yes | Cache the uint8 storage view or CUTE scale tensor with the weight cache; invalidate on weight version change. |
| compile key, major-order metadata, scheduler args for fixed SM100 gated config | static for the model shape/config | yes | Cache with the compiled gated kernel fast path; do not rebuild per microbatch. |
| activation FP8 `x_fp8` | new storage per microbatch | no | Avoid Python dtype-view overhead by passing uint8 storage plus explicit CUTE element type, or create any alias inside the producing C++ op. |
| packed activation scales `x_scales_tk_pre.view(E8M0)` | new storage and new data per DeepEP dispatch/microbatch | no | Prefer returning an E8M0-typed tensor directly from the metadata C++ op, or keep uint8 storage and pass `Float8E8M0FNU` explicitly to CUTE. |
| z/y1 output FP8 tensors | new storage per microbatch and saved for backward/downstream | no | Allocate/alias at the microbatch boundary, ideally through one C++ carrier factory or a lifetime-aware workspace. |
| z/y1 scale outputs | new storage per microbatch and consumed by backward/downstream | no | Store as uint8 plus explicit scale type, or create the typed alias once next to allocation; do not recreate it in the GEMM prelaunch path. |
| lightweight bf16 zero-stride placeholders | graph/lifetime tied to the current autograd microbatch | no by default | Can be optimized only with an autograd-aware placeholder/cache test; not a P0 compared with B/view/CUTE wrapper costs. |

Therefore the most useful split is:

1. **Global-step hoist:** static weight views, static scale views, B major-order
   metadata, compile key, scheduler args, and any static CUTE wrappers whose
   data pointer is stable for the whole optimizer step.
2. **Microbatch-level elimination:** dynamic activation/output/scale aliases
   cannot be reused globally, but their Python `view_dtype` cost can be removed
   by using typed C++ outputs or a CUTE helper that accepts raw uint8 storage with
   an explicit FP8/E8M0 element type.
3. **Carrier allocation:** `empty` for z/y1 carriers cannot be globally reused
   without autograd lifetime proof.  The safe optimization is either a C++ carrier
   factory that returns all required tensors in one call, or a ring/workspace
   keyed by in-flight microbatch lifetime.

## Existing Reproduction Commands

Use the ERNIEBot venv when reproducing in the integrated environment:

```bash
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/erniebot/venv/bin/activate
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/erniebot/third_party/PaddleFleet/packages/paddlefleet_ops/third_party/sonic-moe
export SONIC_MOE_QUACK_PATH=/root/paddlejob/share-storage/gpfs/system-public/panzhaowu/baidu/ernie/erniebot/third_party/PaddleFleet/packages/paddlefleet_ops/third_party/quack
```

Correctness:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_deepep_topk_metadata.py -q
```

Scale-pack NCU benchmark:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --repeat 100

ncu --set full --launch-skip 10 --launch-count 2 \
  -o /tmp/ncu_topk_prod_a35b_default \
  python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --repeat 1
```

Alternative scale-pack variants:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --scatter-pack --repeat 100
CUDA_VISIBLE_DEVICES=0 python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --row-pack --repeat 100
```

Current measured scale-pack result:

| path | avg GPU time | NCU kernel duration | decision |
|---|---:|---:|---|
| default tile-pack, 256 threads | ~0.309 ms | 231.7 us | keep |
| default tile-pack, 512 threads | ~0.305 ms | not clearly better | not worth changing default |
| default tile-pack, 1024 threads | ~1.213 ms | much worse | reject |
| scatter-pack inside `scatter_and_fixup` | ~0.663 ms | 627.6 us | reject |
| row-major load + shared transpose pack | ~0.778 ms | 710.9 us | reject |

## Experiment And Test Matrix

| priority | experiment | hypothesis | implementation target | reproduction / unit test | performance gate | correctness gate | status |
|---:|---|---|---|---|---|---|---|
| P0 | Verify fused metadata+scale path is active in real A35B training. | Removing the old `scatter->scale` host gap and separate scale kernel should recover about `275-310 us` p50 before the up GEMM. | Fleet call site must call `deepep_topk_to_sonic_metadata_with_scales()` and pass packed scales as the third `x_fp8_pre` item into Sonic. | `tests/ops/test_deepep_topk_metadata.py::TestCudaScalePacking::test_with_scales_matches_raw_gather_reference`; then short A35B nsys. | Timeline must show `pack_raw_scales_from_gather_kernel<...>` and no `_gather_raw_scales_1x32_to_isa_kernel` on the up path. | Packed scale bytes must match `gather_raw_blockscaled_1x32_scales_to_isa()` for all tested shapes/dtypes. | Implemented locally, but latest analyzed production timeline did not hit it. |
| P0 | Keep default tile-pack scale kernel. | Scale-pack kernel tuning alone cannot remove the bubble; alternate packing strategies are slower. | `deepep_topk_metadata_cuda_with_scales` default path. | `python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --repeat 100`; NCU full report. | Default remains around `0.31 ms` event time for `prod_a35b`; scatter/rowpack must not become default unless they beat it. | Same scale reference test as above. | Done; keep default. |
| P1 | Cache static B permute view for varlen up GEMM. | `B.permute(1,2,0)` is a view-only operation but costs about `50-60 us` p50 in wrapper time; B is static per layer. | FP8 weight/scales cache plus `gemm_gated` fast path that accepts already-permuted B/layout metadata. | Add a small/medium shape test comparing generic `gemm_gated` and cached-permute fast path output bytes for z, y1, z scales, and y1 scales. | Same-thread `transpose` p50 in the prelaunch window should approach zero. | Bit-exact output for repeated forward calls; fallback must trigger for unsupported layout. | Proposed. |
| P1 | Remove repeated scale `view_dtype` wrappers. | Packed scales can stay as `uint8` storage while CUTE receives the intended FP8/E8M0 element type explicitly. | Add a helper like `make_cute_tensor_dynamic_fp8_storage(tensor_uint8, element_type, leading_dim)` and use it for packed scale carriers. | Add tests that compare old `.view(_E8M0_DTYPE)` path vs uint8-storage path for small, non-multiple, and A35B-like scale shapes. | Same-thread `view_dtype` p50 should drop by a large fraction of `70-85 us`. | Packed bytes and Sonic GEMM outputs must be bit-exact. | Proposed. |
| P1 | Avoid repeated large carrier construction where lifetime allows. | `empty`/allocator wrapper cost is about `60 us` p50, but the carriers are huge and live through downstream/backward. | Autograd/lifetime-aware workspace, keyed by shape/stream/layer or owned by saved context; never a single naive global scratch. | Add forward+backward lifetime tests: repeated microbatches, delayed backward, and explicit pointer reuse assertions that no live saved tensor is overwritten. | Same-thread `empty` p50 should drop; no increase in peak memory beyond the configured workspace budget. | Forward/backward bit-exact vs old path; no `data_ptr` instability for persistent params/grad buffers. | Proposed; high memory risk. |
| P1 | Add specialized SM100 zero-materialized gated launch fast path. | The remaining `~250-300 us` uninstrumented prelaunch time is generic Python/CUTE wrapper work. | Bypass repeated validation, major-order discovery, `permute_tensors`, compile-key construction, and redundant epilogue/scheduler/varlen object construction for the known production config; keep generic fallback. | Add shape/config matrix comparing generic and fast path bit-exact outputs; include unsupported shape fallback tests. | `scale_pack end -> up GEMM launch API start` p50 should fall materially below `~485 us` after P0/P1 view fixes. | Bit-exact z/y1/scales; repeated calls must use the cached compiled kernel safely. | Proposed; highest wrapper payoff. |
| P2 | Queue metadata conversion and up GEMM behind one host entry point. | The host should enqueue metadata and GEMM before the GPU reaches the boundary, eliminating most visible idle. | C++/CUTE extension wrapper or equivalent Python-free launch path for metadata + up GEMM. | Add stream-order tests with sentinels and compare against separated launches; run nsys gap check. | Target the full `scale_pack -> up GEMM` bubble after P0/P1. | Bit-exact outputs and no cross-stream ordering bugs. | Larger integration change. |
| P2 | Revisit persistent layout/copy movement around optimizer. | Earlier full-run gap had large D2D copy bursts and optimizer-step regression even after layout kernels were reduced. | Fleet/Sonic persistent buffer lifecycle, not this metadata kernel alone. | 8-node short run with parent log rank-max optimizer-step and CUPTI D2D copy extraction. | D2D bytes and optimizer-step rank-max should approach baseline. | Persistent parameter/main_grad/param.grad `data_ptr()` must remain stable. | Separate Fleet/Sonic integration work. |

## PR Acceptance Gates

Before claiming the metadata/prelaunch bubble is fixed:

| gate | required evidence |
|---|---|
| Unit correctness | `tests/ops/test_deepep_topk_metadata.py` passes, especially `TestCudaScalePacking`. |
| Scale-pack identity | Packed scales match `gather_raw_blockscaled_1x32_scales_to_isa()` byte-for-byte for int32 and uint8 raw scales. |
| Production-path activation | A35B nsys shows the fused kernel name and no old `_gather_raw_scales_1x32_to_isa_kernel` in the up path. |
| Prelaunch gap | Same-launch-thread SQL shows p50 `scale_pack end -> up GEMM launch API start` reduced from the current `~485 us`. |
| End-to-end sanity | A35B short run keeps barrier-on correctness and does not regress loss/norm against the known A1B/35B baseline policy. |
| No pointer breakage | Persistent params, `main_grad`, and `param.grad` retain `data_ptr()` stability; zero-copy sharing is allowed only for temporary carriers with proven lifetime. |

## Validation Run For This Document

Run on 2026-07-09 from the ERNIEBot venv in the integrated checkout:

| check | command | result |
|---|---|---|
| scale-pack bit-exact subset | `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_deepep_topk_metadata.py::TestCudaScalePacking -q --tb=short` | `9 passed` |
| full DeepEP topk metadata tests | `CUDA_VISIBLE_DEVICES=0 python -m pytest tests/ops/test_deepep_topk_metadata.py -q --tb=short` | `74 passed` |
| production-shape scale-pack harness | `CUDA_VISIBLE_DEVICES=0 python tests/ops/ncu_deepep_topk.py --config prod_a35b --with-scales --repeat 10` | `avg_gpu_ms=0.356854`, `packed_scales_shape=(1, 208044032)` |

## Non-Goals

- Do not disable `moe_ep_barrier` to recover time.
- Do not center this gap on MoE allgather; the current profile uses DeepEP.
- Do not spend more time micro-tuning `_gather_raw_scales_1x32_to_isa_kernel`
  alone.  It is about `13 us` p50; the expensive part is host-side sequencing
  and wrapper/carrier construction around the up GEMM launch.
- Do not introduce a global scratch carrier without autograd lifetime proof.
