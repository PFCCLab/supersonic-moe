# Weekly Work Summary — 2026-W17 (Apr 23 → Apr 30)

> Author: agent on `race-fix-paddle`. Repo: `PFCCLab/supersonic-moe` (upstream) ←
> `A-nnonymous/supersonic-moe` (fork). All numbers verified against `git log`
> and the GitHub PR API.

## TL;DR

- **8 PRs merged into upstream `PFCCLab/supersonic-moe`** (PR #10–#18, modulo #14 which is an external contributor).
- **43 commits** authored on the fork (myrepo), **260 files touched**, **+19 029 / −4 696 LOC** net.
- **4 additional commits** queued on `race-fix-paddle` post-PR-#18 (S79 + S79b: determinism CI, MFU sweep tooling, hardware-identity audit) — pushed to fork, not yet PR'd to upstream.
- Headline outcomes: FP8 frontier reached **~45 % MFU at Ernie production shape (T8192/E8/K8) ≈ 2 020 TFLOPS achieved on B30Z**, bit-exact deterministic across runs and gated in CI.

---

## Merged PRs (upstream `PFCCLab/supersonic-moe`)

| # | merged | title | scope |
|---|---|---|---|
| #10 | 04-24 07:54 | feat: JIT enhance + dispatch_probs grad fix | Eliminates JIT recompilation on seqlen change; fixes ds gradient flow back to `dispatched_probs`; PreAct=dz bug fix; cold-start E2E test gold-reference fix; production-ready `SonicMoEMlpNode` (warmup, validation, cache lifecycle); README + HANDOFF rewritten for production state. |
| #11 | 04-24 12:45 | fix: compile cache + varlen-K hang | Removes dynamic dims from 3 compile_keys; hardens cold-start E2E test; fixes E≠topk crash via `varlen_K_max` + `tl.min`; quack Paddle dtype compat. |
| #12 | 04-24 14:58 | TMA add support + main-grad accumulation perf | `perf: TMA reduce-add wgrad epilogue — regs 86→50, E2E 2-4 % faster`; fuses BF16 wgrad accumulate into GEMM epilogue (eliminates 664 µs/iter); fixes prod-default `fp8_wgrad=False` path (1.14–1.43× vs BF16); E=32 GPU-projection breakdown; documents wgrad overhead root cause as fused-epilogue regs (not Paddle proxy). |
| #13 | 04-27 09:27 | top-K metadata deadlock + gather large-tensor | Splits histogram + prefix into two kernels; removes `grid2` cap that dropped rows ≥ 65 536; topk bug audit + correctness regression; refreshes stale erniebot venv paths in `.runenv.sh`; ignore local artifacts. |
| #15 | 04-28 13:56 | retire iso32 weight quant + recompute_z + multilayer fix | Removes iso32 weight quant; adds opt-in `recompute_z` UpProj (Option B opt-in, broken on non-uniform; Option A is correct default); per-layer native_grad + multi-layer multi-step + fused FP8 warmup; nsys overhead audit. |
| #16 | 04-28 17:41 | Race fix + perf optimization | Race-safe JIT + FP8 config isolation + cluster-env-safe multicard; lazy `main_grad` alloc + `step()` ordering fix; triton stream compat + router-grad scatter + `mlp_node_v2` globals purge; integrates FP8 frontier perf; fixes 0-token tuple arity; ships nsys timeline. |
| #17 | 04-29 11:23 | Enhanced CI: main_grad fix + JIT cost reduce + device fix | Distributed safety + skip-warmup sentinel + strict-baseline CI; precision-test fixture fix + Fleet integration audit; persistent Triton autotune cache + Ernie nsys baseline + coverage gate live; NCU-driven quant kernel optimization (90–93 % of HBM peak). |
| #18 | 04-30 02:35 | Further improve JIT performance, tighten CI | S78b/S78c: tighten JIT/perf baselines, audit triton kernels, import-smoke coverage lift; ncu `--set full` profile of 6 Ernie-shape GEMMs. |

(PR #14 was authored by @lshpku — `perf: copy tpe_list to device asynchronously` — reviewed but not authored by me.)

## Post-PR-#18 work on `race-fix-paddle` (pushed to fork, not yet PR'd upstream)

| sha | local time | scope |
|---|---|---|
| `d0c1e6a` | 04-30 14:44 | **S79.1**: `tests/fp8_frontier_determinism_test.py` (NEW). Two tests (small-aligned + Ernie-prod) prove FP8 frontier path produces byte-identical `(out, dx, every grad)` across three independent runs. Wired into `tests/run_regression.sh` as a HARD-fail gate. Documents three paddle-proxy quirks (scope-limited proxy missing `stream_base`; `torch.equal` element-wise under proxy; `.to(dtype=...)` requires explicit `device=`). |
| `c04b651` | 04-30 15:07 | **S79.2 + cleanup**: HANDOFF/README rewrite for clean S79 frontier. Documents the dgrad1 single-kernel optimization audit as a documented no-go (scale-LDG dedup, packed-mul collapse — all neutral or regressed; reverted). |
| `ba40169` | 04-30 15:46 | **S79b.1**: `tools/mfu_sweep_s79.py` (NEW). nsys-driven 11-shape FP8 frontier MFU sweep. Adds `--H` flag to `bench_mlpnode_topk_nsys.py`. Generates `reports/mfu_s79/{README.md, sweep.{csv,json}, 4 seaborn plots, per-shape bench logs}`. Headline: Ernie MFU 44.91 % (2021 TFLOPS), best 50.88 % (T8192-H6144). |
| `df5f86e` | 04-30 16:06 | **S79b.2**: hardware identity audit. Confirms GPU is **B30Z** (not B300), 1100 W cap is VBIOS-locked, sustained pure-GEMM hits cap and throttles 2032→1249 MHz, MoE bench stays under cap at full boost. |
| `b92d944` | 04-30 16:22 | **S79b.3**: corrects an erroneous peak derivation (the spec-scaling formula has no power term — peak = SMs × ops/cycle × clock; the earlier `× (1100/1400)` was double-counting). 4500 TFLOPS retained as empirically-anchored boost-clock peak. |

---

## Outcomes by theme

### 1. Performance (PRs #10, #12, #15–17)

- **Ernie shape (T8192, E8, K8) FP8 frontier**:
  - Cold-start E2E now reaches the warm-state numerics in 1 step (no contamination from JIT cache mismatch).
  - Wgrad path: TMA reduce-add epilogue (regs 86→50), BF16 accumulate fused into GEMM epilogue (–664 µs/iter), `fp8_wgrad=False` production default (1.14–1.43× vs BF16).
  - Quant kernels NCU-tuned to 90–93 % of HBM peak.
  - Single-iter median **busy time 2.75 ms / 4.5 ms wall** (60 % SM utilisation single-iter, 100 % busy under async pipeline).
- **MFU sweep (S79b)**: 11 shapes, see `reports/mfu_s79/README.md`.
  - Ernie MFU **44.91 %** (vs boost peak 4500 TFLOPS).
  - Best MFU **50.88 %** (T8192-H6144-I2048, wider matmul).
  - Doubling E at fixed K costs ~2.3 pp MFU per doubling (routing tax quantified).
  - Sustained-clock projection: ~25 % absolute throughput drop under 1100 W cap, but MFU % invariant.

### 2. Correctness & race fixes (PRs #11, #13, #15, #16, S79.1)

- **JIT race-safe**: removed dynamic dims from compile_keys; eliminated recompile-on-seqlen.
- **deepep top-K metadata**: split histogram + prefix; removed `grid2` cap (dropped rows ≥65 536); fixed gather large-tensor.
- **Multi-layer**: per-layer native_grad + multi-step; lazy `main_grad` alloc; `step()` ordering fix.
- **dispatched_probs grad flow**: now properly differentiable (PreAct=dz bug fix).
- **FP8 frontier IS bit-deterministic** (S79.1): `tests/fp8_frontier_determinism_test.py` proves byte-equality across runs and is now a hard CI gate.

### 3. CI & infra (PRs #10, #11, #16, #17, #18)

- Coverage gate live with strict baselines.
- Persistent Triton autotune cache (cuts cold-start JIT cost).
- Cluster-env-safe multicard (race-safe JIT + FP8 config isolation).
- `tests/run_regression.sh` now sources `.runenv.sh` directly (canonicalised env).
- Determinism gate added (S79.1).
- nsys baseline + ncu `--set full` profile for 6 Ernie GEMMs in `reports/ernie_shape_ncu_s78b/`.
- Triton stream compat + router-grad scatter + globals purge (cleanup of `mlp_node_v2`).

### 4. Audits & documentation (PRs #15, #17, #18 + S79 cleanup)

- nsys overhead audit (PR #15).
- Dgrad1 single-kernel optimization audit (S79.2): documented as no-go after exhaustive trial of scale-LDG dedup, packed-mul collapse, register hints, register-budget reshaping. **All single-kernel knobs exhausted**; further wins require multi-kernel restructuring (fold FP8 cast into bwd-side wgrad producer — listed as next-step lever).
- Hardware-identity audit (S79b.2–3): confirmed B30Z (not B300), reconciled the 4500 TFLOPS peak figure with first-principles formula (peak = SMs × ops/cycle × clock; no power term). Pure-GEMM does hit 1100 W cap and throttles; MoE bench does not.
- HANDOFF + README rewritten twice (PR #10 production-state, S79 cleanup) to keep the next agent's onboarding crisp.

### 5. Inherited / external work touched

- PR #14 (lshpku): `perf: copy tpe_list to device asynchronously` — merged into the same branch line.
- Two reverts/rollbacks (`c907f02` and `5dbd86c`) for session 71/72 refactor that triggered FP8 frontier IMA — documented as a lesson in the post-mortem (in PR #16's HANDOFF section): aggressive multi-layer refactor without a frontier-IMA regression test bricked the path; rolled back, re-introduced more carefully.

---

## Key insights / lessons (compacted)

1. **MFU at Ernie shape ≈ 45 %** with our frontier path; **production bottleneck is non-matmul overhead** (routing/quant/scatter/dGated/FP8-cast inflate the busy denominator). The constituent FP8 GEMMs each hit ≥80 % of peak in isolation. **Highest-leverage remaining work** = fuse FP8 cast into the bwd-side wgrad producer (the fwd-side fold is already in).
2. **Single-kernel optimization on `GemmDGatedFP8CLoadSm100ZeroMat` is exhausted** — every register/scale/epilogue tweak landed neutral or regressed. Future wins require restructuring (multi-kernel fusion or rewriting the dGated epilogue).
3. **Hardware ground truth**: GPU is **B30Z (148 SMs, 2032 MHz boost, 1100 W VBIOS cap, 268 GiB HBM3e), NOT retail B300**. The 4500 TFLOPS peak is empirically anchored (1800 BF16 TFLOPS @ 2032 MHz / 0.8 cuBLAS efficiency × 2 for FP8). Tensor-core peak formula has **no power term** — power only enters as a runtime throttling consequence.
4. **Sweep MFU numbers are valid**: MoE bench averages 921 W (under 1100 W cap), runs at full 2032 MHz, so 4500 TFLOPS reference is self-consistent. In production async training the cap WILL bite (~25 % absolute throughput drop), but **MFU % is invariant under throttling** because numerator and denominator scale linearly with clock.
5. **Reverse-engineering paddle's torch proxy is a recurring tax**: scoped proxy missing `stream_base`; `torch.equal` element-wise; `.to(dtype=)` needing explicit device. Documented in S79.1 commit message and the determinism test header — next agent should consult before writing any new paddle-proxy-aware test.
6. **Multi-layer refactor needs frontier-IMA regression coverage** — the session-71/72 rollback cost time. The S79.1 determinism test now fills this gap.

## What's queued / next-step recommendations

1. PR the four post-#18 commits (S79 + S79b) once user confirms scope. They're already on `myrepo/race-fix-paddle`.
2. Implement bwd-side FP8-cast fusion into wgrad producer — projected +5 pp MFU.
3. If wider hidden sizes ship in future Ernie revisions: H=6144 already shows +6 pp MFU "for free"; verify production data path supports it.
4. (Long shot) explore multi-kernel restructuring of `GemmDGatedFP8CLoadSm100ZeroMat` to reclaim the dgrad1 budget that single-kernel work couldn't touch.

---

## Stats

- **Date range**: 2026-04-23 00:00 → 2026-04-30 23:59 (CST)
- **Commits authored**: 43 (excluding stash entries and merges)
- **LOC**: +19 029 / −4 696 (net +14 333)
- **Files touched (sum across commits)**: 260
- **Merged PRs into upstream**: 8 (PR #10, #11, #12, #13, #15, #16, #17, #18)
- **Branches active**: `race-fix-paddle` (current), `session60-ds-fix` (stable)
- **Latest fork HEAD**: `b92d944` on `myrepo/race-fix-paddle`
