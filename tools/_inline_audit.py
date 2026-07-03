"""Inline audit: monkey-patch backward to print tensor inventory at dgated completion."""
import sys, os, gc, torch
sys.path.insert(0, os.environ.get("SONIC_MOE_REPO", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ["USE_QUACK_GEMM"] = "1"
os.environ["SONIC_MOE_FP8_MODE"] = "perf"
torch.manual_seed(42)
MiB = 1024**2
T, H, I, E, K = 8192, 3072, 1536, 8, 8

from sonicmoe import MoE
from sonicmoe.enums import ActivationType
from sonicmoe.functional.utils import enable_quack_gemm

moe = MoE(num_experts=E, num_experts_per_tok=K, hidden_size=H, intermediate_size=I,
    activation_function=ActivationType.SWIGLU, add_bias=False, std=0.02).to(device="cuda", dtype=torch.bfloat16)
x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16, requires_grad=True)
dout = torch.randn(T, H, device="cuda", dtype=torch.bfloat16)

for _ in range(2):
    with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
    o.backward(dout); x.grad = None
    for p in moe.parameters():
        if p.grad is not None: p.grad = None

moe.refresh_fp8_shadow_weights()
gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()

# Use memory_allocated_bytes at fine granularity via hooks
_prev_alloc = [0]
_audit_log = []

def _alloc_hook(device, alloc_size):
    """Track every allocation > 1 MiB."""
    current = torch.cuda.memory_allocated()
    if abs(alloc_size) > 1 * MiB:
        _audit_log.append((current / MiB, alloc_size / MiB, len(_audit_log)))

# Can't easily hook allocations. Instead, use memory_stats diff approach.
# Just measure before and after each known operation in backward.

torch.cuda.reset_peak_memory_stats()
with enable_quack_gemm(True): o = moe(x, use_fp8=True)[0]
torch.cuda.synchronize()
fwd_peak = torch.cuda.max_memory_allocated() / MiB

torch.cuda.reset_peak_memory_stats()
o.backward(dout)
torch.cuda.synchronize()
bwd_peak = torch.cuda.max_memory_allocated() / MiB

# Report memory_stats breakdown
stats = torch.cuda.memory_stats()
print(f"=== CUDA MEMORY STATS ===")
print(f"Fwd peak alloc: {fwd_peak:.1f} MiB")
print(f"Bwd peak alloc: {bwd_peak:.1f} MiB")
print(f"Current alloc:  {torch.cuda.memory_allocated()/MiB:.1f} MiB")
print(f"Current reserved: {torch.cuda.memory_reserved()/MiB:.1f} MiB")
print()

# Key stats
for key in sorted(stats.keys()):
    if 'peak' in key or 'allocated' in key.lower():
        val = stats[key]
        if isinstance(val, int) and val > 1000:
            print(f"  {key}: {val / MiB:.1f} MiB" if val > MiB else f"  {key}: {val}")

# What we KNOW is in the peak
print(f"\n=== THEORETICAL PEAK BREAKDOWN (backward:dgated completion) ===")
items = [
    ("ctx: x (saved)", 48.0, "bf16 (T,H)"),
    ("ctx: w1 (param ref)", 0, "shared with model"),
    ("ctx: w2 (param ref)", 0, "shared with model"),
    ("ctx: z_fp8", 192.0, "fp8 (TK,2I)"),
    ("ctx: z_raw_scales", 6.0, "e8m0"),
    ("ctx: topk_scores", 0.25, "fp32 (TK,)"),
    ("ctx: expert_freq_offset", 0.0, "int32 (E+1)"),
    ("ctx: x_gather_idx", 0.25, "int32 (TK,)"),
    ("ctx: s_scatter_idx", 0.25, "int32 (TK,)"),
    ("ctx: s_reverse_scatter_idx", 0.25, "int32 (TK,)"),
    ("model: w1", 144.0, "bf16 param"),
    ("model: w2", 72.0, "bf16 param"),
    ("model: router_w", 0.05, "bf16 param"),
    ("input: dout", 48.0, "bf16 (T,H)"),
    ("cache: w1T varlen fp8", 72.0, "fp8 (E,H,2I)"),
    ("cache: w1T varlen scales", 2.25, "e8m0"),
    ("cache: w2 fused fp8", 36.0, "fp8 (E,I,H)"),
    ("cache: w2 fused scales", 1.12, "e8m0"),
    ("dgated output: dz", 384.0, "bf16 (TK,2I) ← BIGGEST"),
    ("dgated output: y1s", 192.0, "bf16 (TK,I)"),
    ("dgated pre-alloc: dw2_base", 0.0, "DEFERRED to wgrad (session 42)"),
    ("dgated input: dout_fp8", 24.0, "fp8 (T,H)"),
    ("dgated input: dout_scales", 6.0, "e8m0 ISA TK"),
    ("dgated input: w2_fp8_enk", 36.0, "fp8 (E,I,H) — from cache"),
    ("dgated input: w2_scales", 1.12, "e8m0"),
    ("dgated output: colvec_reduce", 10.0, "fp32 (TK,ceil)"),
    ("dgated: s_float", 0.25, "fp32 (TK,)"),
    ("dgated: config", 0, "python object"),
    ("routing: s", 0.5, "bf16 (TK,)"),
    ("autograd graph overhead", 20.0, "estimate"),
]

total = sum(x[1] for x in items)
print(f"{'Item':<40s} {'Size':>8s}  Note")
print("-" * 80)
for name, size, note in items:
    if size > 0.01:
        print(f"  {name:<38s} {size:>6.1f}M  {note}")
print(f"\n  {'TOTAL THEORETICAL':38s} {total:>6.1f}M")
print(f"  {'MEASURED PEAK':38s} {bwd_peak:>6.1f}M")
print(f"  {'GAP (unaccounted)':38s} {bwd_peak - total:>6.1f}M ({(bwd_peak-total)/bwd_peak*100:.1f}%)")
