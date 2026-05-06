# Directory Index: `/tests/`

> Regression, integration, and contract tests for SonicMoE.

## How to run (eb_venv — runs all tests)

```bash
EBVENV=/root/paddlejob/share-storage/gpfs/system-public/zhangyichen/erniebot/eb_venv
export USE_QUACK_GEMM=1 SONIC_MOE_FP8_ASSUME_ALIGNED=1

# Quantization + SwiGLU tests (fully passing)
CUDA_VISIBLE_DEVICES=0 $EBVENV/bin/python -m pytest tests/ops/test_rowwise_quant.py tests/ops/test_colwise_quant.py tests/ops/test_dequant.py tests/ops/test_dual_quant.py tests/ops/test_fused_zy1_quant.py tests/ops/test_weight_quant.py tests/ops/test_swiglu.py -q

# MoE module (fully passing)
CUDA_VISIBLE_DEVICES=1 $EBVENV/bin/python -m pytest tests/ops/test_moe_module.py -q

# FP8 routing + main_grad (Paddle compat script, fully passing)
CUDA_VISIBLE_DEVICES=2 $EBVENV/bin/python tests/ops/test_moe_general_routing_fp8.py

# ERNIE-compat SonicMoEFunc (Paddle compat script, fully passing)
CUDA_VISIBLE_DEVICES=3 $EBVENV/bin/python tests/ops/test_sonic_moe_func.py

# GEMM tests (BLOCKED by paddle compat stream_base issue)
# CUDA_VISIBLE_DEVICES=4 $EBVENV/bin/python -m pytest tests/ops/test_gemm_gated.py tests/ops/test_gemm_dgated.py tests/ops/test_wgrad_gemm.py -q

# Varlen + pad tests (BLOCKED by paddle compat _is_in_bad_fork issue)
# CUDA_VISIBLE_DEVICES=5 $EBVENV/bin/python -m pytest tests/ops/test_varlen_gemm.py tests/ops/test_pad_routing.py tests/ops/test_pad_gradient_integrity.py -q
```

### Known blocked tests (paddle compat gaps, not test bugs)

| Tests | Failure | Root cause |
|-------|---------|------------|
| gemm_gated, gemm_dgated, wgrad_gemm | `'Stream' has no attribute 'stream_base'` | Paddle compat Stream wrapper incomplete |
| varlen_gemm, pad_routing, pad_gradient | `paddle.cuda has no '_is_in_bad_fork'` | Paddle compat missing torch.cuda internal API |

These tests pass in a pure-torch environment (xfer) but require `sonicmoe/__init__.py` to support optional paddle import first.

## Child directories

| Path | Summary |
| --- | --- |
| `ops/` | Focused operator and module-level tests. |

## Files — regression suite

| File | What it tests | Status |
| --- | --- | --- |
| `test_commons.py` | Shared `TestCommons` base class | infra |
| `fp8_operator_options.py` | Shared `OperatorOpt` config | infra |
| `moe_test.py` | Full MoE fwd+bwd combinatorics | needs xfer |
| `moe_blackwell_test.py` | Blackwell SM100 smoke test | needs xfer |
| `fp8_protocol_test.py` | FP8Protocol API coverage | needs xfer |
| `fp8_large_project_contract_test.py` | FP8 contract E=128 | needs xfer |
| `fp8_frontier_strict_test.py` | Strict fused-gated test | needs xfer |
| `fp8_frontier_determinism_test.py` | Bit-exact determinism on FP8 frontier (fused-gated, alignment_assumed) | runs in CI (eb_venv via .runenv.sh) |
| `test_blockscaled_fp8_varlen.py` | Varlen GEMM against gold | needs xfer |
| `count_cumsum_test.py` | count_cumsum extension | needs xfer |
| `run_regression.sh` | Regression runner | needs xfer |
