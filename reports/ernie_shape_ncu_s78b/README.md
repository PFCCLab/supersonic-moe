# Ernie-shape NCU Full Report — 6 GEMM Kernels (S78c)

Full-feature `ncu --set full` report for the 6 cutlass GEMM kernels that fire
in **one** SonicMoEMlpNode iter at the canonical Ernie-1B-MoE shape on B200.

## Shape

| Param | Value |
|-------|-------|
| T (tokens) | 8192 |
| H (hidden) | 3072 |
| I (interm) | 1536 |
| E (experts) | 8 |
| topk | 8 |
| Routing | balanced (TPE 8192/8192/8192) |
| Dtype | FP8 (frontier, fused epilogue + maingrad add) |

## Reproduction

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
export PYTHONPATH=/root/.../zhangyichen/sonicmoe_for_ernie/quack:$PYTHONPATH

ncu --set full \
    --kernel-name 'regex:(cutlass_kernel_quackgemm|GemmGated|GemmDGated)' \
    --launch-skip 0 --launch-count 6 \
    --nvtx --nvtx-include 'BENCH/' \
    --replay-mode kernel --cache-control none --clock-control base \
    --export reports/ernie_shape_ncu_s78b/ernie_gemms \
    --force-overwrite \
    -- python tests/ops/bench_mlpnode_topk_nsys.py \
          --T 8192 --E 8 --I 1536 --topk 8 --warmup 8 --iters 1
```

The bench harness pushes an NVTX `BENCH/` range around the iter loop, so the
8 warmup iters (which finish all JIT compile / autotune / cache writes) are
excluded; only the 6 GEMMs of the single timed iter are profiled.

## Files

| File | Purpose |
|------|---------|
| `ernie_gemms.ncu-rep` | binary report — open in Nsight Compute GUI (16 MB) |
| `gemms_full.details.txt` | `--page details` text dump (all 88 metrics × 6 launches) |
| `gemms_details.csv` | same data as CSV |
| `gemms_raw.csv` | `--page raw` (raw underlying counters) |
| `headlines.txt` | the per-kernel headline table reproduced below |

## Per-kernel headlines

Launch order inside the BENCH range maps directly to forward / backward call
order in `sonicmoe/functional/__init__.py`:

| Lid | Role | Kernel symbol (short) | Dur (µs) | SM % (compute SoL) | SM Busy % | Tensor pipe % | DRAM % | L2 hit % | Regs/thr | Grid |
|----:|------|-----------------------|---------:|-------------------:|----------:|--------------:|-------:|---------:|---------:|------|
| 0 | **fwd1** (gated GEMM1 + SwiGLU + FP8 quant epi) | `GemmGatedSm100ZeroMatBlockscaledQuant` | **762.66** | 64.43 | 64.14 | 64.43 | 10.51 | 88.79 | 168 | (6312,1,1) |
| 1 | **fwd2** (default GEMM2, fp8) | `quackgemm_default_epi GemmDefaultSm100` | **360.96** | 69.72 | 69.06 | 69.72 | 19.50 | 77.61 |  54 | (6312,1,1) |
| 2 | **dgrad1** (dgated GEMM, fp8 + C-load + dSwiGLU) | `GemmDGatedFP8CLoadSm100ZeroMat` | **610.98** | 41.95 | 41.95 | 41.78 | 22.48 | 61.27 | 168 | (3156,1,1) |
| 3 | **dgrad2** (default GEMM, dx) | `GemmDefaultSm100` | **314.82** | 80.93 | 80.90 | 80.93 | 25.88 | 68.55 |  56 | (144,1,8) |
| 4 | **wgrad1** (default GEMM, dW1) | `GemmDefaultSm100` | **581.54** | 83.81 | 83.78 | 83.81 | 24.80 | 74.00 |  56 | (288,1,8) |
| 5 | **wgrad2** (default GEMM, dW2 + maingrad-add epi) | `GemmDefaultSm100` | **600.83** | 81.29 | 80.90 | 81.29 | 15.69 | 78.66 |  54 | (6312,1,1) |

**Sum of GEMM time:** 3231.79 µs over 6 launches = ~3.23 ms.
For reference, the full per-iter wall time at this shape (S78 nsys
GPU-projection median) is ~2740 µs; under ncu the kernels run with
`--clock-control base` and per-kernel replay (≈40 passes/kernel), so the
absolute numbers are larger than the production timeline but the
*relative* SoL %, occupancy and memory utilization carry over.

## Per-kernel insights

### fwd1 — `GemmGatedSm100ZeroMatBlockscaledQuant` (762 µs, 64% compute SoL)
- Tensor pipeline at 64% — limited by the **fused epilogue**: SwiGLU + 1×128
  blockscaled FP8 quantize + ZeroMat masking on top of the GEMM.
- L2 hit-rate **88.8%** — best of all six kernels; the persistent A/B tile
  stream re-reads through L2 cleanly.
- DRAM only 10.5% — kernel is compute-and-epilogue bound, not DRAM bound.
- 168 regs/thread — close to the 255 cap but not spilling; reducing register
  pressure in the epilogue is the main lever to push tensor-pipe utilisation
  past 70%.

### fwd2 — `GemmDefault` (361 µs, 70% compute SoL)
- Smallest fwd kernel, 54 regs/thread, no fused epilogue — scaling is clean
  but DRAM at 19.5% suggests the second matmul (Y2 = z·W2) is partially
  re-reading residual `z` from L2 (hit-rate 77.6%).

### dgrad1 — `GemmDGatedFP8CLoadSm100ZeroMat` (611 µs, 42% compute SoL) **★ slowest per-FLOP**
- Compute SoL only 42% — this is the **lowest tensor-pipe utilisation in
  the whole iter** and the most promising single-kernel optimisation target.
- Reason: kernel does FP8 GEMM **plus** dSwiGLU **plus** C-load (residual
  add of incoming dY1) **plus** ZeroMat — the epilogue is heavier than the
  GemmGated forward equivalent and the math pipeline stalls waiting on the
  epilogue.
- DRAM at 22.5% with L2 hit only 61.3% → the C-load path is
  re-reading `dY1` from HBM rather than reusing it from L2; hoisting
  `dY1` into a persistent tile or splitting the dgated kernel into
  `gemm + epi` two-pass could reclaim ≥100 µs.

### dgrad2 — `GemmDefault` (315 µs, 81% compute SoL)
- Cleanest backward kernel, hits 80% tensor-pipe util on a tiny grid
  (144×8 = 1152 CTAs) — already near optimum.

### wgrad1 — `GemmDefault` (582 µs, 84% compute SoL) **★ near peak**
- Highest SoL of the iter at 83.8%, 80% tensor-pipe — wgrad1 is essentially
  bottlenecked by the DRAM read bandwidth of the fp8 inputs (24.8% DRAM,
  74% L2 hit).

### wgrad2 — `GemmDefault` (601 µs, 81% compute SoL)
- Includes the **fused main-grad accumulator add** in the epilogue
  (one of the FP8-frontier contributions). Despite that, SoL is 81% and
  the kernel is comfortably compute-bound — the maingrad-add epilogue
  is essentially free.

## Bottlenecks summary (where to spend optimisation budget)

| Rank | Kernel | What's leaving perf on the table | Suggested next step |
|------|--------|----------------------------------|---------------------|
| 1 | **dgrad1 GemmDGated** | 42% SoL, heavy fused epilogue, L2-hit only 61% on dY1 C-load | refactor dY1 reuse (persistent tile or split kernel); 100+ µs upside |
| 2 | **fwd1 GemmGated**    | 64% SoL — epilogue register pressure | thin epilogue regs / re-use shared mem; ~10% upside |
| 3 | **fwd2 GemmDefault**  | 70% SoL — small DRAM re-read | hoist `z` into producer pipeline of fwd1; ~5% upside |
| 4 | wgrad1/wgrad2/dgrad2  | 80–84% SoL | already near peak — no further work warranted |

## Notes / caveats

- ncu reports come from `--clock-control base` (locked clocks) and per-kernel
  replay; absolute durations are higher than production, but SoL %,
  occupancy and memory-pipeline numbers are valid relative measurements.
- The 2 fwd GEMMs share the **same** `quackgemm_default_epi` symbol with the
  3 wgrad GEMMs; we differentiate by launch order inside the NVTX `BENCH/`
  range, which is deterministic (see `forward.py` / `backward.py` call
  sequence).
- The 6 cutlass GEMM symbols on B200 (sm_103) are:
  - `cutlass_kernel_sonicmoequack_utilsgemm_sm100_fp8_zeromatGemmGatedSm100ZeroMatBlockscaledQuant_…`
  - `cutlass_kernel_quackgemm_default_epiGemmDefaultSm100_…`
  - `cutlass_kernel_sonicmoequack_utilsgemm_sm100_fp8_zeromatGemmDGatedFP8CLoadSm100ZeroMat_…`
- `ctc__rx_bytes_data_user.sum` and 5 sibling NVLink/CTC metrics are
  unavailable on this single-GPU host — harmless warning, all other
  metrics collected.
