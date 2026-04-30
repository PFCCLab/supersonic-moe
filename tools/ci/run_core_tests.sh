#!/usr/bin/env bash
# sonic-moe core regression runner.
#
# Strict-baseline CI: every core mechanism is measured and gated.
# Phase coverage (✓ = always run, F = full only):
#
#   ✓  precision        tests/ops/test_mlpnode_precision.py (script, 6 shapes)
#   ✓  multilayer       tests/ops/test_mlpnode_multilayer.py (PP+multistep)
#   F  quant            full quant kernel sweep
#   F  jit-cold         tools/ci/jit_bench.py --phase cold     (budget in baselines.json)
#   F  jit-warm         tools/ci/jit_bench.py --phase warm     (sentinel skip <5s)
#   F  jit-reload       tools/ci/jit_bench.py --phase reload   (disk-cache reload)
#   F  jit-reuse        tools/ci/jit_bench.py --phase reuse    (in-proc dispatch)
#   F  perf             tools/ci/perf_gate.py  (nsys GPU-projection)
#   F  multicard        tools/ci/multicard_smoke.py (auto-skip on single GPU)
#   ✓  coverage         coverage report (omits cli/ + *_compat.py)
#
# Usage:
#   tools/ci/run_core_tests.sh                  # full sweep
#   tools/ci/run_core_tests.sh --fast           # precision + multilayer + jit-warm + coverage
#   tools/ci/run_core_tests.sh --no-coverage --no-perf --no-multicard --no-jit
#   tools/ci/run_core_tests.sh --soft-jit       # warn-only on JIT budgets
#   tools/ci/run_core_tests.sh --baselines path/to/custom.json
#
# Exit codes:
#   0 on full success; 1 on any phase failure; 2 on bad args.
set -u
set -o pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BASELINES="$ROOT/tools/ci/baselines.json"
FAST=0; DO_COVERAGE=1; DO_PERF=1; DO_MULTICARD=1; DO_JIT=1; SOFT_JIT=0
for arg in "$@"; do
  case "$arg" in
    --fast)         FAST=1; DO_PERF=0; DO_MULTICARD=0 ;;  # JIT-warm still runs
    --no-coverage)  DO_COVERAGE=0 ;;
    --no-perf)      DO_PERF=0 ;;
    --no-multicard) DO_MULTICARD=0 ;;
    --no-jit)       DO_JIT=0 ;;
    --soft-jit)     SOFT_JIT=1 ;;
    --baselines=*)  BASELINES="${arg#*=}" ;;
    --baselines)    shift; BASELINES="$1" ;;
    -h|--help)
      grep -E '^# ?' "$0" | sed -e 's/^# \?//' -e '/^!/d'
      exit 0 ;;
    *)
      echo "[ci] unknown arg: $arg" >&2; exit 2 ;;
  esac
done

[[ -f "$BASELINES" ]] || { echo "[ci] missing baselines: $BASELINES" >&2; exit 2; }

CACHE_ROOT="${SONIC_MOE_CI_CACHE_DIR:-$ROOT/.jit_cache_ci}"
export SONIC_MOE_CACHE_DIR="$CACHE_ROOT"
mkdir -p "$ROOT/.ci_artifacts"
JIT_JSON="$ROOT/.ci_artifacts/jit_bench.json"

GREEN=$'\033[32m'; RED=$'\033[31m'; YEL=$'\033[33m'; RST=$'\033[0m'
PASS_ROWS=(); FAIL_ROWS=(); SKIP_ROWS=()

phase() {
  local name="$1"; shift
  echo
  echo "${YEL}── [${name}] ──${RST}  $*"
  local t0; t0="$(date +%s)"
  if "$@"; then
    local dt=$(( $(date +%s) - t0 ))
    echo "${GREEN}[${name}] PASS (${dt}s)${RST}"
    PASS_ROWS+=("${name} (${dt}s)")
  else
    local rc=$? dt=$(( $(date +%s) - t0 ))
    echo "${RED}[${name}] FAIL rc=${rc} (${dt}s)${RST}"
    FAIL_ROWS+=("${name} rc=${rc}")
  fi
}

skip() { echo "${YEL}[${1}] SKIP — ${2}${RST}"; SKIP_ROWS+=("${1}: ${2}"); }

run_pytest() {
  if [[ "$DO_COVERAGE" == "1" ]] && command -v coverage >/dev/null 2>&1; then
    coverage run --source=sonicmoe -m pytest "$@" 2>&1 | tail -200
  else
    python -m pytest "$@" 2>&1 | tail -200
  fi
}

# Parallel pytest helper. Each xdist worker is pinned to a distinct GPU by
# tests/conftest.py BEFORE any cuda init. Coverage is collected per worker
# (`coverage` recognises xdist via parallel=True in .coveragerc).
PYTEST_PAR_N="${PYTEST_PAR_N:-}"
if [[ -z "$PYTEST_PAR_N" ]]; then
  GC_PAR="$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 1)"
  PYTEST_PAR_N="${GC_PAR:-1}"
fi
run_pytest_parallel() {
  local n="$PYTEST_PAR_N"
  if (( n < 2 )) || ! python -c "import xdist" 2>/dev/null; then
    run_pytest "$@"
    return
  fi
  if [[ "$DO_COVERAGE" == "1" ]] && command -v coverage >/dev/null 2>&1; then
    COVERAGE_PROCESS_START="$ROOT/.coveragerc" \
      coverage run --source=sonicmoe -m pytest \
        -n "$n" --dist=loadfile "$@" 2>&1 | tail -200
  else
    python -m pytest -n "$n" --dist=loadfile "$@" 2>&1 | tail -200
  fi
}

run_script() {
  if [[ "$DO_COVERAGE" == "1" ]] && command -v coverage >/dev/null 2>&1; then
    coverage run --source=sonicmoe "$@" 2>&1 | tail -200
  else
    python "$@" 2>&1 | tail -200
  fi
}

# ── Coverage init ─────────────────────────────────────────────────────────────
if [[ "$DO_COVERAGE" == "1" ]]; then
  if command -v coverage >/dev/null 2>&1; then
    coverage erase || true
    echo "[ci] coverage enabled (sonicmoe/, .coveragerc)"
  else
    echo "${YEL}[ci] coverage missing — pip install coverage to enable${RST}"
    DO_COVERAGE=0
  fi
fi

# ── 1. Precision ─────────────────────────────────────────────────────────────
phase precision run_script tests/ops/test_mlpnode_precision.py

# ── 1b. Import smoke (cheap; lifts coverage of optional kernels) ─────────────
phase import-smoke run_pytest_parallel tests/ops/test_import_smoke.py -q

# ── 2. Multilayer / PP / multistep ───────────────────────────────────────────
phase multilayer run_pytest_parallel tests/ops/test_mlpnode_multilayer.py -x -q

# ── 3. Quant kernels (full only) ─────────────────────────────────────────────
if [[ "$FAST" == "0" ]]; then
  phase quant run_pytest_parallel \
    tests/ops/test_dual_quant.py \
    tests/ops/test_fused_quant.py \
    tests/ops/test_colwise_quant.py \
    tests/ops/test_rowwise_quant.py \
    tests/ops/test_weight_quant.py \
    tests/ops/test_dequant.py -x -q
else
  skip quant "fast mode"
fi

# ── 4. JIT mechanism baseline (cold + warm + reload + reuse) ─────────────────
if [[ "$DO_JIT" == "1" ]]; then
  JIT_ARGS=(--baselines "$BASELINES" --json "$JIT_JSON")
  [[ "$SOFT_JIT" == "1" ]] && JIT_ARGS+=(--soft)
  if [[ "$FAST" == "1" ]]; then
    # Pre-commit: only the cheap, deterministic indicators (no cold compile,
    # no full reload). 'warm' requires a previously populated cache; if none,
    # fall back to a single 'reuse' check.
    if [[ -f "$CACHE_ROOT/warmup_sentinel.json" ]]; then
      phase jit-warm  python tools/ci/jit_bench.py --phase warm  "${JIT_ARGS[@]}"
      phase jit-reuse python tools/ci/jit_bench.py --phase reuse "${JIT_ARGS[@]}"
    else
      skip jit-warm "no warm cache (run full CI once to populate $CACHE_ROOT)"
      skip jit-reuse "no warm cache"
    fi
  else
    phase jit-cold   python tools/ci/jit_bench.py --phase cold   "${JIT_ARGS[@]}"
    phase jit-warm   python tools/ci/jit_bench.py --phase warm   "${JIT_ARGS[@]}"
    phase jit-reload python tools/ci/jit_bench.py --phase reload "${JIT_ARGS[@]}"
    phase jit-reuse  python tools/ci/jit_bench.py --phase reuse  "${JIT_ARGS[@]}"
    phase jit-parallel python tools/ci/jit_bench.py --phase parallel-cold "${JIT_ARGS[@]}"
  fi
else
  skip jit "--no-jit"
fi

# ── 4b. JIT key stability + extreme-shape robustness ─────────────────────────
phase jit-key-stability run_pytest_parallel tests/ops/test_jit_key_stability.py -x -q
phase extreme-shapes    run_pytest_parallel tests/ops/test_mlpnode_extreme_shapes.py -q
phase jit-concurrent    run_pytest tests/ops/test_jit_concurrent_heterogeneous.py -q

# ── 5. Perf gate (canonical T8192-H3072-I1536-E8-K8) ─────────────────────────
if [[ "$DO_PERF" == "1" ]]; then
  if [[ -x "$(command -v nsys)" ]]; then
    PERF_BUDGET_MS="$(python -c "import json,sys; b=json.load(open('$BASELINES'))['perf']['gpu_projection_us_per_iter']['budget']; print(b/1000.0)")"
    PERF_SHAPE="$(python -c "import json; print(json.load(open('$BASELINES'))['perf']['shape'])")"
    phase perf python tools/ci/perf_gate.py \
      --shape "$PERF_SHAPE" --budget-ms "$PERF_BUDGET_MS"
  else
    skip perf "nsys not in PATH"
  fi
else
  skip perf "--no-perf"
fi

# ── 6. Multi-card slot ───────────────────────────────────────────────────────
if [[ "$DO_MULTICARD" == "1" ]]; then
  GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0)"
  if [[ "${GPU_COUNT:-0}" -ge 2 ]]; then
    phase multicard python tools/ci/multicard_smoke.py
  else
    skip multicard "single-GPU env (count=${GPU_COUNT:-0})"
  fi
else
  skip multicard "--no-multicard"
fi

# ── Coverage report + gate ───────────────────────────────────────────────────
if [[ "$DO_COVERAGE" == "1" ]]; then
  echo
  echo "${YEL}── [coverage] ──${RST}"
  coverage combine 2>/dev/null || true   # merge per-xdist-worker .coverage.*
  coverage report --skip-covered 2>&1 | tail -100 || true
  coverage html -d "$ROOT/.coverage_html" 2>/dev/null && \
    echo "[ci] HTML report: $ROOT/.coverage_html/index.html"
  COVERAGE_TARGET="$(python -c "import json; print(json.load(open('$BASELINES')).get('coverage',{}).get('target_pct', 0))")"
  if [[ "${COVERAGE_TARGET:-0}" -gt 0 ]]; then
    if coverage report --fail-under="$COVERAGE_TARGET" >/dev/null 2>&1; then
      echo "${GREEN}[coverage-gate] PASS (≥${COVERAGE_TARGET}%)${RST}"
      PASS_ROWS+=("coverage-gate (≥${COVERAGE_TARGET}%)")
    else
      ACTUAL="$(coverage report 2>/dev/null | tail -1 | awk '{print $NF}')"
      echo "${RED}[coverage-gate] FAIL — got ${ACTUAL}, need ≥${COVERAGE_TARGET}%${RST}"
      FAIL_ROWS+=("coverage-gate ${ACTUAL}<${COVERAGE_TARGET}%")
    fi
  fi
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo
echo "═════════════════ CI SUMMARY ═════════════════"
for r in "${PASS_ROWS[@]}"; do echo "${GREEN}PASS${RST} ${r}"; done
for r in "${SKIP_ROWS[@]}"; do echo "${YEL}SKIP${RST} ${r}"; done
for r in "${FAIL_ROWS[@]}"; do echo "${RED}FAIL${RST} ${r}"; done
echo "──────────────────────────────────────────────"
[[ -f "$JIT_JSON" ]] && echo "JIT JSON: $JIT_JSON"
echo "Baselines: $BASELINES"
echo "Cache root: $CACHE_ROOT"
echo "══════════════════════════════════════════════"

(( ${#FAIL_ROWS[@]} == 0 )) || exit 1
exit 0
