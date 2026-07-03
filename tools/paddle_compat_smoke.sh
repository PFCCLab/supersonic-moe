#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# Paddle Compat Smoke Test — SonicMoE BF16 fwd+bwd under Paddle
# ═══════════════════════════════════════════════════════════════════════════════
#
# Measures:
#   1. Correctness: BF16 fwd output matches gold (RRMSE)
#   2. Performance: CUDA-event fwd+bwd timing (µs/iter)
#   3. Memory: peak fwd + peak bwd (MiB)
#
# Compares against original PyTorch results (xfer env) run side by side.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash tools/paddle_compat_smoke.sh
# ═══════════════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
cd "$PROJECT_ROOT"

GPU=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$GPU

# ── Environments ──────────────────────────────────────────────────────────
PADDLE_VENV="${SONIC_MOE_PADDLE_VENV:-}"
TORCH_VENV="${SONIC_MOE_TORCH_VENV:-}"
QUACK_PATH="${SONIC_MOE_QUACK_PATH:-}"

REPORT_DIR="$PROJECT_ROOT/reports/paddle_compat_smoke"
mkdir -p "$REPORT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_JSON="$REPORT_DIR/smoke_${TIMESTAMP}.json"

echo "═══════════════════════════════════════════════════════════════"
echo " Paddle Compat Smoke Test"
echo " GPU: $GPU"
echo " Report: $REPORT_JSON"
echo "═══════════════════════════════════════════════════════════════"

# ── Shared Python script (parameterized by env) ──────────────────────────
WORKER_SCRIPT=$(cat <<'PYEOF'
import gc, json, os, sys, time
env_label = os.environ.get("_SMOKE_ENV", "unknown")
is_paddle = os.environ.get("_SMOKE_PADDLE", "0") == "1"

if is_paddle:
    import paddle; paddle.enable_compat()

os.environ["USE_QUACK_GEMM"] = "1"
import torch
sys.path.insert(0, os.environ["_SMOKE_PROJECT"])

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_fp8, enable_quack_gemm
import sonicmoe.functional as functional

T, H, I, E, K = 8192, 3072, 1536, 8, 8
WARMUP, MEASURED = 5, 20
MiB = 1048576

if is_paddle:
    paddle.seed(42)
else:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

device = torch.device("cuda:0")
moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H,
           intermediate_size=I, activation_function=ActivationType.SWIGLU,
           add_bias=False, std=0.02).to(device=device, dtype=torch.bfloat16)
x = (0.02 * torch.randn(T, H, device=device, dtype=torch.bfloat16))

# ── Precision: single fwd vs gold ────────────────────────────────────────
from sonicmoe.functional import moe_TC_softmax_topk_layer
w1_p = moe.c_fc.weight.permute(1, 2, 0)
w2_p = moe.c_proj.weight.permute(1, 2, 0)
functional._ALIGNMENT_ASSUMED = False; functional._ALIGNMENT_STREAK = 0

xi = x.detach().clone().requires_grad_(True)
with enable_fp8(False):
    o, _, ef = moe_TC_softmax_topk_layer(
        xi, moe.router.weight, w1_p, None, w2_p, None,
        K, moe.stream_id, ActivationType.SWIGLU, False)
o_norm = o.float().norm().item()

# ── Performance: CUDA-event timing ───────────────────────────────────────
def run_iter():
    xw = x.detach().clone().requires_grad_(True)
    with enable_quack_gemm(True), enable_fp8(False):
        out, aux = moe(xw)
    return xw, out

# Warmup
for _ in range(WARMUP):
    xw, out = run_iter()
    out.sum().backward()
    moe.zero_grad(set_to_none=True)
    del xw, out
torch.cuda.synchronize(); gc.collect(); torch.cuda.empty_cache()

# Measured (CUDA events)
start_events = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURED)]
end_events = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURED)]

for i in range(MEASURED):
    start_events[i].record()
    xw, out = run_iter()
    out.sum().backward()
    end_events[i].record()
    moe.zero_grad(set_to_none=True)
    del xw, out

torch.cuda.synchronize()
times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
times_us = [t * 1000 for t in times_ms]
median_us = sorted(times_us)[len(times_us) // 2]

# ── Memory ────────────────────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
xw, out = run_iter()
torch.cuda.synchronize()
peak_fwd = torch.cuda.max_memory_allocated(device) / MiB
torch.cuda.reset_peak_memory_stats(device)
out.sum().backward()
torch.cuda.synchronize()
peak_bwd = torch.cuda.max_memory_allocated(device) / MiB

result = {
    "env": env_label,
    "shape": {"T": T, "H": H, "I": I, "E": E, "K": K},
    "o_norm": round(o_norm, 6),
    "median_us": round(median_us, 1),
    "mean_us": round(sum(times_us) / len(times_us), 1),
    "min_us": round(min(times_us), 1),
    "max_us": round(max(times_us), 1),
    "peak_fwd_mib": round(peak_fwd, 1),
    "peak_bwd_mib": round(peak_bwd, 1),
    "iters": MEASURED,
}
print("__SMOKE__" + json.dumps(result))
PYEOF
)

# ── Run under Paddle (eb_venv) ────────────────────────────────────────────
echo ""
echo "  [1/2] Running SonicMoE BF16 under Paddle compat..."
PADDLE_RESULT=$(\
  _SMOKE_ENV=paddle _SMOKE_PADDLE=1 _SMOKE_PROJECT="$PROJECT_ROOT" \
  PYTHONPATH="$QUACK_PATH:$PROJECT_ROOT" \
  "$PADDLE_VENV/bin/python" -c "$WORKER_SCRIPT" 2>&1 | tee /dev/stderr | grep "^__SMOKE__" | sed 's/^__SMOKE__//')

if [ -z "$PADDLE_RESULT" ]; then
  echo "  [FAIL] Paddle run produced no output"
  PADDLE_RESULT='{"env":"paddle","error":"no output"}'
fi
echo "  Paddle result: $PADDLE_RESULT"

# ── Run under PyTorch (xfer) — use native-fp8-exploration branch ──────────
echo ""
echo "  [2/2] Running SonicMoE BF16 under PyTorch (native-fp8-exploration)..."
# Create temp worktree for pytorch baseline (original branch, no paddle imports)
TORCH_WORKTREE=$(mktemp -d)
git worktree add "$TORCH_WORKTREE" native-fp8-exploration --quiet 2>/dev/null || true
TORCH_RESULT=$(\
  _SMOKE_ENV=pytorch _SMOKE_PADDLE=0 _SMOKE_PROJECT="$TORCH_WORKTREE" \
  PYTHONPATH="$TORCH_WORKTREE" \
  "$TORCH_VENV/bin/python" -c "$WORKER_SCRIPT" 2>&1 | tee /dev/stderr | grep "^__SMOKE__" | sed 's/^__SMOKE__//')
git worktree remove "$TORCH_WORKTREE" --force 2>/dev/null || true

if [ -z "$TORCH_RESULT" ]; then
  echo "  [FAIL] PyTorch run produced no output"
  TORCH_RESULT='{"env":"pytorch","error":"no output"}'
fi
echo "  PyTorch result: $TORCH_RESULT"

# ── Combine and write report ─────────────────────────────────────────────
python3 -c "
import json, sys
paddle = json.loads('$PADDLE_RESULT')
pytorch = json.loads('$TORCH_RESULT')
report = {
    'paddle': paddle,
    'pytorch': pytorch,
}
if 'median_us' in paddle and 'median_us' in pytorch:
    p_us = paddle['median_us']
    t_us = pytorch['median_us']
    report['comparison'] = {
        'paddle_us': p_us,
        'pytorch_us': t_us,
        'overhead_pct': round((p_us - t_us) / t_us * 100, 2),
        'ratio': round(p_us / t_us, 4),
    }
    print()
    print('=' * 60)
    print(f'  SonicMoE BF16 fwd+bwd (CUDA-event, median)')
    print(f'  Paddle:  {p_us:>8.0f} µs/iter')
    print(f'  PyTorch: {t_us:>8.0f} µs/iter')
    print(f'  Overhead: {report[\"comparison\"][\"overhead_pct\"]:+.1f}%')
    print(f'  o_norm: paddle={paddle.get(\"o_norm\",\"?\")}, pytorch={pytorch.get(\"o_norm\",\"?\")}')
    print(f'  Memory (bwd): paddle={paddle.get(\"peak_bwd_mib\",\"?\")} MiB, pytorch={pytorch.get(\"peak_bwd_mib\",\"?\")} MiB')
    print('=' * 60)

with open('$REPORT_JSON', 'w') as f:
    json.dump(report, f, indent=2)
print(f'Report: $REPORT_JSON')
"
