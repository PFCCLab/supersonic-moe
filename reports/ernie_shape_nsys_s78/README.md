# Ernie-shape nsys timeline — S78

> Captured 2026-04-29 on B30Z (paddlejob shared-GPFS host) via the same
> CI helper that gates `tools/ci/run_core_tests.sh::perf` —
> `tools/ci/perf_gate.py --shape T8192-H3072-I1536-E8-K8 --keep-trace`.
> Underlying bench: `tests/ops/bench_mlpnode_topk_nsys.py` (FP8 frontier,
> `SonicMoEMlpNode` fwd+bwd, fused epilogue main_grad add, FP8 wgrad).

## Files

| File | Purpose |
| ---- | ------- |
| `trace.nsys-rep` | Open in Nsight Systems 2026.2 GUI/CLI. |
| `trace.sqlite`   | nsys-export sqlite — feed to `tools/parse_nsys_per_iter.py` or to `tests/ops/bench_mlpnode_topk_nsys.py --extract`. |
| `breakdown.txt`  | Top-N per-kernel µs/iter restricted to the NVTX `BENCH` range (excludes warmup/JIT/autotune). |

## Run parameters

* Shape: `T=8192 H=3072 I=1536 E=8 topk=8` (TK=65536)
* Imbalance: `none` (uniform `tokens_per_expert=8192`)
* Warmup: 8 iters
* Bench: 12 iters, NVTX-tagged `BENCH`
* Path: FP8 frontier (`enable_fp8(True)`) — fused epilogue main_grad add,
  FP8 wgrad, dual-quant, gated SM100 GEMM, sonic-moe Triton stream patch
* GPU: NVIDIA B30Z (compute capability 10.3, driver 13.0)
* CUDA stream: paddle compute stream (Triton patch verified active)

## Headline numbers

| Metric | Value |
| ------ | -----: |
| GPU-projection (busy-time sum / iters) | **2740.1 µs/iter** |
| GPU-projection (min..max span / iters) | 5106.4 µs/iter |
| CPU wall inside `BENCH` / iters        | 5137.6 µs/iter |
| Perf-gate budget                       | 4500.0 µs/iter |
| % of budget                            | **60.9 %** ✅ |
| S53 PyTorch native FP8 baseline        | 2715 µs/iter |
| Δ vs. S53                              | +25 µs (+0.9 %) |

The (busy-time / span) ratio = 2740 / 5106 ≈ **53.7 % SM utilisation**, i.e. the
remaining 46 % is pure stream-gap (kernel-launch latency + paddle dispatch +
the deepep-metadata path). This matches the S77 HANDOFF projection.

## Per-iter kernel breakdown (BENCH window, 12 iters)

| Kernel | n/iter | µs/iter | % of GPU busy |
| ------ | -----: | ------: | ------------: |
| `kernel_cutlass_kernel_quackgemm_default_epi*` (fwd up + down GEMM)        | 4 | 1215.8 | 44.4 % |
| `kernel_cutlass_…_zeromatGemmGated*` (FP8 dgrad, gated)                    | 1 |  449.8 | 16.4 % |
| `kernel_cutlass_…_zeromatGemmDGated*` (FP8 wgrad, gated)                   | 1 |  399.3 | 14.6 % |
| `_colwise_quantize_and_pack_kernel`                                         | 3 |  213.9 |  7.8 % |
| `_dual_varlen_quantize_kernel`                                              | 1 |  152.4 |  5.6 % |
| `token_gather_sum_kernel`                                                   | 2 |  147.9 |  5.4 % |
| `_quantize_and_pack_kernel`                                                 | 3 |   80.9 |  3.0 % |
| `VectorizedBroadcastKernel`                                                 | 1 |   19.3 |  0.7 % |
| `index_elementwise_with_tensor_kernel`                                      | 3 |   16.6 |  0.6 % |
| `_gather_isa_packed_scales_kernel`                                          | 2 |   15.8 |  0.6 % |
| `VectorizedElementwiseKernel`                                               | 6 |   10.7 |  0.4 % |
| `_scatter_router_grad_kernel` (router-scores backward — S74 fix)            | 1 |    6.3 |  0.2 % |
| miscellaneous (`VecReduceKernel`, histogram, prefix_sums, …)                | – |    9.4 |  0.3 % |
| **TOTAL (busy)**                                                            | – | **2740.1** | 100 % |

Notes on what is **absent** (verifies the S74 stream + router fixes):

* No `cub::DeviceRadixSort*` cascade behind the router-scores backward —
  replaced by the single `_scatter_router_grad_kernel` line above.
* No sonic-moe Triton kernel on the legacy NULL stream (verified via
  `streamId` join — every quant/gather/scatter kernel is on stream 13,
  same as `kernel_cutlass_*`). Triton stream patch active.

## Reproduce

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export PYTHONPATH=/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/sonicmoe_for_ernie/quack:${PYTHONPATH:-}
python tools/ci/perf_gate.py --shape T8192-H3072-I1536-E8-K8 \
    --budget-ms 4.5 --iters 12 --warmup 8 --keep-trace
# trace.nsys-rep / trace.sqlite land in /tmp/sonicmoe_ci_perf/
```
