# Directory Index: `/tools/ci/`

> Bash-driven CI runners and gates for sonic-moe core regressions.
> Local-first; same scripts can be wired into GitHub Actions or pre-commit.

## Maintenance rules
- Update this `INDEX.md` whenever scripts are added / renamed / restructured.
- Each script must remain runnable from the repository root with no extra setup beyond `source .runenv.sh` (or equivalent).

## Files
| File | Summary | Notes |
| --- | --- | --- |
| `baselines.json` | Strict per-mechanism budgets (cold/warm/reload/reuse JIT + nsys perf gate) consumed by `run_core_tests.sh` and `jit_bench.py`. Every loosening must be justified in HANDOFF. | S76 |
| `run_core_tests.sh` | Top-level runner: precision script, multilayer pytest, quant sweep, JIT mechanism gates (`jit_bench.py`), nsys perf gate, multi-card smoke, coverage report. `--fast` for pre-commit. | S76 |
| `jit_bench.py` | Strict 4-axis JIT mechanism gate: cold-warmup / warm-sentinel-skip / cross-process-disk-reload / in-process-reuse — each in a fresh subprocess, results enforced against `baselines.json`. Resilient to known `paddle.utils.hipify` env bug (reports SKIP, not FAIL). | S76 |
| `perf_gate.py` | nsys-driven per-iter GPU-projection check on canonical T8192-H3072-I1536-E8-K8 vs configurable budget. | S76 |
| `multicard_smoke.py` | Spawns `paddle.distributed.launch --gpus 0,1` and runs a 2-rank `SonicMoEMlpNode` fwd+bwd; rank-0 verifies finite + cosine. Skipped automatically when fewer than 2 GPUs visible. | S76 |
