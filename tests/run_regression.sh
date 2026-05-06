#!/bin/bash
set -e
source /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe/.runenv.sh
cd /root/paddlejob/share-storage/gpfs/system-public/panzhaowu/lab/sonic-moe

echo "=== Running full regression test suite ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo ""

# Core regression tests
echo "--- fp8_large_project_contract_test.py (31 tests) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_large_project_contract_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- fp8_protocol_test.py (26 tests) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_protocol_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- moe_blackwell_test.py ---"
USE_QUACK_GEMM=1 python -m pytest tests/moe_blackwell_test.py -v --tb=short 2>&1 || true

echo ""
echo "--- moe_test.py ---"
USE_QUACK_GEMM=1 python -m pytest tests/moe_test.py -v --tb=short 2>&1 || true

# FP8 frontier determinism — HARD failure (non-determinism is a correctness bug,
# not a flaky test). Do NOT swallow with `|| true`.
echo ""
echo "--- fp8_frontier_determinism_test.py (2 tests, HARD-fail on non-determinism) ---"
USE_QUACK_GEMM=1 python -m pytest tests/fp8_frontier_determinism_test.py -v --tb=short 2>&1

echo ""
echo "=== Regression complete ==="
