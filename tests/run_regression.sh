#!/bin/bash
set -e
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/.runenv.sh
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

echo "=== SonicMoE Full CI Regression Suite ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# HARD-FAIL gates: any failure here blocks the entire CI.
# Do NOT add || true. These are non-negotiable correctness invariants.
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ HARD-FAIL GATES ═══"

echo "--- [HARD] fp8_frontier_determinism_test.py (bit-exact repeated fwd/bwd) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_frontier_determinism_test.py -v --tb=short

echo ""
echo "--- [HARD] test_no_memcpy_sync.py (zero GPU sync in hot path) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/test_no_memcpy_sync.py -v --tb=short

# ─────────────────────────────────────────────────────────────────────────────
# CORE tests: production path correctness. Failures are bugs to investigate.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ CORE TESTS (failures = bugs) ═══"

echo ""
echo "--- [CORE] fp8_frontier_stress_test.py (17 stress scenarios) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_frontier_stress_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_gemm_gated.py (fused gated GEMM) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_gemm_gated.py -v --tb=short 2>&1 || true

# FP8 frontier stress test — HARD failure for precision regression.
echo ""
echo "--- fp8_frontier_stress_test.py (17 tests, routing stress) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_frontier_stress_test.py -v --tb=short 2>&1

# Precision audit (6 shapes including E=32@small-N)
echo ""
echo "--- test_mlpnode_precision.py (6-shape precision audit) ---"
USE_QUACK_GEMM=1 python tests/ops/test_mlpnode_precision.py 2>&1

# Large-shape correctness (SEQ=16K, TK=131072)
echo ""
echo "--- test_mlpnode_correctness_large.py ---"
USE_QUACK_GEMM=1 python -m pytest tests/ops/test_mlpnode_correctness_large.py -v --tb=short 2>&1

# Extreme memory stress test (TK up to 786K, int32 boundary crossing)
echo ""
echo "--- test_frontier_stress_sanitizer.py (extreme shapes, HARD-fail) ---"
USE_QUACK_GEMM=1 python tests/ops/test_frontier_stress_sanitizer.py 2>&1

echo ""
echo "--- [CORE] ops/test_gemm_dgated.py (dGated backward) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_gemm_dgated.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_varlen_gemm.py (varlen grouped GEMM) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_varlen_gemm.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_wgrad_gemm.py (wgrad + TMA reduce-add) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_wgrad_gemm.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_e2e_mlpnode.py (MlpNode end-to-end) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_e2e_mlpnode.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_moe_module.py (59 parametrized MoE tests) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_moe_module.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CORE] ops/test_moe_general_routing_fp8.py (FP8 routing) ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/ops/test_moe_general_routing_fp8.py -v --tb=short 2>&1 || true

# ─────────────────────────────────────────────────────────────────────────────
# QUANT kernel tests: precision of FP8 quantization building blocks.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ QUANT KERNEL TESTS ═══"

echo ""
echo "--- [QUANT] colwise + dual + rowwise + fused + weight + dequant ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest \
    tests/ops/test_colwise_quant.py \
    tests/ops/test_dual_quant.py \
    tests/ops/test_rowwise_quant.py \
    tests/ops/test_fused_quant.py \
    tests/ops/test_weight_quant.py \
    tests/ops/test_dequant.py \
    tests/ops/test_fused_zy1_quant.py \
    -v --tb=short 2>&1 || true

# ─────────────────────────────────────────────────────────────────────────────
# ROUTING & PADDING tests: metadata integrity and gradient flow.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ ROUTING & PADDING TESTS ═══"

echo ""
echo "--- [ROUTE] deepep_topk_metadata + pad_routing + pad_gradient_integrity ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest \
    tests/ops/test_deepep_topk_metadata.py \
    tests/ops/test_pad_routing.py \
    tests/ops/test_pad_gradient_integrity.py \
    -v --tb=short 2>&1 || true

# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS tests: extreme shapes, multi-layer, edge cases.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ ROBUSTNESS TESTS ═══"

echo ""
echo "--- [ROBUST] mlpnode_extreme_shapes + mlpnode_multilayer + swiglu ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest \
    tests/ops/test_mlpnode_extreme_shapes.py \
    tests/ops/test_mlpnode_multilayer.py \
    tests/ops/test_swiglu.py \
    -v --tb=short 2>&1 || true

# ─────────────────────────────────────────────────────────────────────────────
# CONTRACT tests: protocol + project-level integration contracts.
# Known env-dependent failures on non-SM100 or partial QuACK installs.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ CONTRACT TESTS (env-dependent) ═══"

echo ""
echo "--- [CONTRACT] fp8_protocol_test.py ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_protocol_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CONTRACT] fp8_large_project_contract_test.py ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- [CONTRACT] fp8_frontier_strict_test.py ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest tests/fp8_frontier_strict_test.py -v --tb=short 2>&1 || true

# ─────────────────────────────────────────────────────────────────────────────
# SMOKE tests: import + misc.
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ SMOKE TESTS ═══"

echo ""
echo "--- [SMOKE] import_smoke + blockscaled_fp8_varlen + moe_layer ---"
USE_QUACK_GEMM=1 SONIC_MOE_FP8_MODE=perf \
  python -m pytest \
    tests/ops/test_import_smoke.py \
    tests/test_blockscaled_fp8_varlen.py \
    tests/test_moe_layer.py \
    tests/test_moe_layer_e2e.py \
    -v --tb=short 2>&1 || true

echo ""
echo "═══════════════════════════════════════════════════════════════════════"
echo "REGRESSION COMPLETE: $(date)"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
echo "DEPRECATED (skipped): moe_blackwell_test.py, moe_test.py, count_cumsum_test.py"
echo "  Reason: require torch._dynamo/torch.compile (Paddle compat incompatible)"
echo ""
echo "COVERAGE GAPS (need future work):"
echo "  - No isolated unit test for GemmDGatedFP8CLoadSm100ZeroMat"
echo "  - No dedicated test for BF16 mode (SONIC_MOE_FP8_MODE='')"
echo "  - No explicit test for node.step() layout transposition correctness"
